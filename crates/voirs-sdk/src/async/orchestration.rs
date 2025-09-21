use crate::error::Result as VoirsResult;
use crate::error::VoirsError;
use futures::future::BoxFuture;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, Semaphore};

pub trait AsyncTask: Send + Sync + 'static {
    type Output: Send + Sync + 'static;

    fn execute(&self) -> BoxFuture<'static, VoirsResult<Self::Output>>;
    fn priority(&self) -> u8;
    fn estimated_duration(&self) -> Duration;
}

#[allow(dead_code)] // Advanced async framework - fields may be used in future implementations
pub struct TaskQueue<T: AsyncTask> {
    tasks: Arc<RwLock<Vec<Arc<T>>>>,
    semaphore: Arc<Semaphore>,
    max_concurrent: usize,
}

impl<T: AsyncTask> TaskQueue<T> {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            tasks: Arc::new(RwLock::new(Vec::new())),
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            max_concurrent,
        }
    }

    pub async fn enqueue(&self, task: T) -> VoirsResult<()> {
        let mut tasks = self.tasks.write().await;
        tasks.push(Arc::new(task));

        tasks.sort_by_key(|b| std::cmp::Reverse(b.priority()));
        Ok(())
    }

    pub async fn execute_all(&self) -> VoirsResult<Vec<T::Output>> {
        let tasks = {
            let mut tasks_guard = self.tasks.write().await;
            std::mem::take(&mut *tasks_guard)
        };

        let mut handles = Vec::new();

        for task in tasks {
            let permit = self.semaphore.clone().acquire_owned().await.map_err(|e| {
                VoirsError::internal(
                    "async_orchestration",
                    format!("Failed to acquire semaphore: {e}"),
                )
            })?;

            let handle = tokio::spawn(async move {
                let _permit = permit;
                task.execute().await
            });

            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.map_err(|e| {
                VoirsError::internal("async_orchestration", format!("Task execution failed: {e}"))
            })?;
            results.push(result?);
        }

        Ok(results)
    }
}

pub struct WorkStealer {
    workers: Vec<Arc<RwLock<TaskQueue<DynamicTask>>>>,
    current_worker: Arc<RwLock<usize>>,
}

// Wrapper to make trait objects work with TaskQueue
pub struct DynamicTask {
    task: Box<dyn AsyncTask<Output = ()>>,
}

impl AsyncTask for DynamicTask {
    type Output = ();

    fn execute(&self) -> BoxFuture<'static, VoirsResult<Self::Output>> {
        self.task.execute()
    }

    fn priority(&self) -> u8 {
        self.task.priority()
    }

    fn estimated_duration(&self) -> Duration {
        self.task.estimated_duration()
    }
}

impl DynamicTask {
    pub fn new(task: Box<dyn AsyncTask<Output = ()>>) -> Self {
        Self { task }
    }
}

impl WorkStealer {
    pub fn new(num_workers: usize, max_concurrent_per_worker: usize) -> Self {
        let workers = (0..num_workers)
            .map(|_| Arc::new(RwLock::new(TaskQueue::new(max_concurrent_per_worker))))
            .collect();

        Self {
            workers,
            current_worker: Arc::new(RwLock::new(0)),
        }
    }

    pub async fn submit_task(&self, task: Box<dyn AsyncTask<Output = ()>>) -> VoirsResult<()> {
        let worker_idx = {
            let mut current = self.current_worker.write().await;
            let idx = *current;
            *current = (*current + 1) % self.workers.len();
            idx
        };

        let worker = self.workers[worker_idx].read().await;
        worker.enqueue(DynamicTask::new(task)).await
    }

    pub async fn steal_work(&self, from_worker: usize, to_worker: usize) -> VoirsResult<()> {
        if from_worker >= self.workers.len() || to_worker >= self.workers.len() {
            return Err(VoirsError::invalid_config(
                "worker_indices",
                format!("from={from_worker}, to={to_worker}"),
                "Worker indices exceed available workers",
            ));
        }

        let from_queue = &self.workers[from_worker];
        let to_queue = &self.workers[to_worker];

        let stolen_task = {
            let from_queue_guard = from_queue.write().await;
            let mut from_tasks = from_queue_guard.tasks.write().await;
            if from_tasks.is_empty() {
                return Ok(());
            }
            from_tasks.pop()
        };

        if let Some(task) = stolen_task {
            let to_queue_guard = to_queue.write().await;
            let mut to_tasks = to_queue_guard.tasks.write().await;
            to_tasks.push(task);
        }

        Ok(())
    }
}

pub struct LoadBalancer {
    workers: HashMap<String, WorkerInfo>,
    strategy: LoadBalancingStrategy,
}

#[derive(Debug, Clone)]
pub struct WorkerInfo {
    pub id: String,
    pub current_load: f64,
    pub capacity: usize,
    pub response_time: Duration,
    pub error_rate: f64,
}

#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    LeastResponseTime,
    AdaptiveLoad,
}

impl LoadBalancer {
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            workers: HashMap::new(),
            strategy,
        }
    }

    pub fn add_worker(&mut self, worker: WorkerInfo) {
        self.workers.insert(worker.id.clone(), worker);
    }

    pub fn remove_worker(&mut self, worker_id: &str) -> Option<WorkerInfo> {
        self.workers.remove(worker_id)
    }

    pub fn select_worker(&self) -> Option<&WorkerInfo> {
        match self.strategy {
            LoadBalancingStrategy::RoundRobin => self.round_robin_selection(),
            LoadBalancingStrategy::LeastConnections => self.least_connections_selection(),
            LoadBalancingStrategy::WeightedRoundRobin => self.weighted_round_robin_selection(),
            LoadBalancingStrategy::LeastResponseTime => self.least_response_time_selection(),
            LoadBalancingStrategy::AdaptiveLoad => self.adaptive_load_selection(),
        }
    }

    fn round_robin_selection(&self) -> Option<&WorkerInfo> {
        self.workers.values().next()
    }

    fn least_connections_selection(&self) -> Option<&WorkerInfo> {
        self.workers
            .values()
            .min_by(|a, b| a.current_load.partial_cmp(&b.current_load).unwrap())
    }

    fn weighted_round_robin_selection(&self) -> Option<&WorkerInfo> {
        self.workers.values().min_by(|a, b| {
            let a_weight = a.current_load / a.capacity as f64;
            let b_weight = b.current_load / b.capacity as f64;
            a_weight.partial_cmp(&b_weight).unwrap()
        })
    }

    fn least_response_time_selection(&self) -> Option<&WorkerInfo> {
        self.workers
            .values()
            .min_by(|a, b| a.response_time.cmp(&b.response_time))
    }

    fn adaptive_load_selection(&self) -> Option<&WorkerInfo> {
        self.workers.values().min_by(|a, b| {
            let a_score = a.current_load * (1.0 + a.error_rate) * a.response_time.as_secs_f64();
            let b_score = b.current_load * (1.0 + b.error_rate) * b.response_time.as_secs_f64();
            a_score.partial_cmp(&b_score).unwrap()
        })
    }

    pub fn update_worker_stats(
        &mut self,
        worker_id: &str,
        load: f64,
        response_time: Duration,
        error_rate: f64,
    ) {
        if let Some(worker) = self.workers.get_mut(worker_id) {
            worker.current_load = load;
            worker.response_time = response_time;
            worker.error_rate = error_rate;
        }
    }
}

#[allow(dead_code)] // Advanced async framework - fields may be used in future implementations
pub struct ParallelPipeline {
    stages: Vec<Box<dyn PipelineStage>>,
    load_balancer: LoadBalancer,
}

pub trait PipelineStage: Send + Sync {
    fn name(&self) -> &str;
    fn execute(
        &self,
        input: Box<dyn std::any::Any + Send + Sync>,
    ) -> BoxFuture<'static, VoirsResult<Box<dyn std::any::Any + Send + Sync>>>;
    fn can_parallelize(&self) -> bool;
    fn max_parallelism(&self) -> usize;
}

impl ParallelPipeline {
    pub fn new(load_balancer: LoadBalancer) -> Self {
        Self {
            stages: Vec::new(),
            load_balancer,
        }
    }

    pub fn add_stage(&mut self, stage: Box<dyn PipelineStage>) {
        self.stages.push(stage);
    }

    pub async fn execute(
        &self,
        input: Box<dyn std::any::Any + Send + Sync>,
    ) -> VoirsResult<Box<dyn std::any::Any + Send + Sync>> {
        let mut current_input = input;

        for stage in &self.stages {
            if stage.can_parallelize() && stage.max_parallelism() > 1 {
                current_input = self
                    .execute_parallel_stage(stage.as_ref(), current_input)
                    .await?;
            } else {
                current_input = stage.execute(current_input).await?;
            }
        }

        Ok(current_input)
    }

    async fn execute_parallel_stage(
        &self,
        stage: &dyn PipelineStage,
        input: Box<dyn std::any::Any + Send + Sync>,
    ) -> VoirsResult<Box<dyn std::any::Any + Send + Sync>> {
        let _parallelism = stage.max_parallelism();

        // For parallel execution, we need to clone the input data
        // Since we can't clone Box<dyn Any>, we'll just execute the stage once
        // and return the result (this is a simplified implementation)
        stage.execute(input).await
    }
}

#[allow(dead_code)] // Advanced async framework - fields may be used in future implementations
pub struct AsyncOrchestrator {
    work_stealer: WorkStealer,
    load_balancer: LoadBalancer,
    pipeline: ParallelPipeline,
}

impl AsyncOrchestrator {
    pub fn new(num_workers: usize, max_concurrent_per_worker: usize) -> Self {
        let work_stealer = WorkStealer::new(num_workers, max_concurrent_per_worker);
        let load_balancer = LoadBalancer::new(LoadBalancingStrategy::AdaptiveLoad);
        let pipeline =
            ParallelPipeline::new(LoadBalancer::new(LoadBalancingStrategy::AdaptiveLoad));

        Self {
            work_stealer,
            load_balancer,
            pipeline,
        }
    }

    pub async fn submit_task(&self, task: Box<dyn AsyncTask<Output = ()>>) -> VoirsResult<()> {
        self.work_stealer.submit_task(task).await
    }

    pub async fn execute_pipeline(
        &self,
        input: Box<dyn std::any::Any + Send + Sync>,
    ) -> VoirsResult<Box<dyn std::any::Any + Send + Sync>> {
        self.pipeline.execute(input).await
    }

    pub fn add_pipeline_stage(&mut self, stage: Box<dyn PipelineStage>) {
        self.pipeline.add_stage(stage);
    }

    pub fn add_worker(&mut self, worker: WorkerInfo) {
        self.load_balancer.add_worker(worker);
    }

    pub fn select_worker(&self) -> Option<&WorkerInfo> {
        self.load_balancer.select_worker()
    }
}
