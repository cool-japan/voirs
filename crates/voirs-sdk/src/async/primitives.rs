use crate::error::Result as VoirsResult;
use crate::error::VoirsError;
use futures::future::Future;
use futures::stream::Stream;
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::sync::{watch, Notify, RwLock};
use tokio::time::Sleep;

pub struct CancellationToken {
    is_cancelled: Arc<RwLock<bool>>,
    notify: Arc<Notify>,
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            is_cancelled: Arc::new(RwLock::new(false)),
            notify: Arc::new(Notify::new()),
        }
    }

    pub async fn cancel(&self) {
        let mut is_cancelled = self.is_cancelled.write().await;
        *is_cancelled = true;
        self.notify.notify_waiters();
    }

    pub async fn is_cancelled(&self) -> bool {
        *self.is_cancelled.read().await
    }

    pub async fn wait_for_cancellation(&self) {
        if self.is_cancelled().await {
            return;
        }
        self.notify.notified().await;
    }

    pub fn child_token(&self) -> ChildCancellationToken {
        ChildCancellationToken {
            parent: self.clone(),
            is_cancelled: Arc::new(RwLock::new(false)),
            notify: Arc::new(Notify::new()),
        }
    }
}

impl Clone for CancellationToken {
    fn clone(&self) -> Self {
        Self {
            is_cancelled: Arc::clone(&self.is_cancelled),
            notify: Arc::clone(&self.notify),
        }
    }
}

pub struct ChildCancellationToken {
    parent: CancellationToken,
    is_cancelled: Arc<RwLock<bool>>,
    notify: Arc<Notify>,
}

impl ChildCancellationToken {
    pub async fn cancel(&self) {
        let mut is_cancelled = self.is_cancelled.write().await;
        *is_cancelled = true;
        self.notify.notify_waiters();
    }

    pub async fn is_cancelled(&self) -> bool {
        *self.is_cancelled.read().await || self.parent.is_cancelled().await
    }

    pub async fn wait_for_cancellation(&self) {
        if self.is_cancelled().await {
            return;
        }

        let parent_future = self.parent.wait_for_cancellation();
        let child_future = self.notify.notified();

        tokio::select! {
            _ = parent_future => {}
            _ = child_future => {}
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProgressInfo {
    pub current: u64,
    pub total: u64,
    pub message: String,
    pub started_at: Instant,
    pub estimated_completion: Option<Instant>,
}

impl ProgressInfo {
    pub fn new(total: u64, message: String) -> Self {
        Self {
            current: 0,
            total,
            message,
            started_at: Instant::now(),
            estimated_completion: None,
        }
    }

    pub fn percentage(&self) -> f64 {
        if self.total == 0 {
            return 100.0;
        }
        (self.current as f64 / self.total as f64) * 100.0
    }

    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    pub fn estimated_remaining(&self) -> Option<Duration> {
        if self.current == 0 {
            return None;
        }

        let elapsed = self.elapsed();
        let rate = self.current as f64 / elapsed.as_secs_f64();
        let remaining = self.total - self.current;

        if rate > 0.0 {
            Some(Duration::from_secs_f64(remaining as f64 / rate))
        } else {
            None
        }
    }
}

pub struct ProgressTracker {
    sender: watch::Sender<ProgressInfo>,
    receiver: watch::Receiver<ProgressInfo>,
}

impl ProgressTracker {
    pub fn new(total: u64, message: String) -> Self {
        let initial_progress = ProgressInfo::new(total, message);
        let (sender, receiver) = watch::channel(initial_progress);

        Self { sender, receiver }
    }

    pub async fn update(&self, current: u64, message: Option<String>) -> VoirsResult<()> {
        let mut progress = self.receiver.borrow().clone();
        progress.current = current;

        if let Some(msg) = message {
            progress.message = msg;
        }

        if let Some(remaining) = progress.estimated_remaining() {
            progress.estimated_completion = Some(Instant::now() + remaining);
        }

        self.sender.send(progress).map_err(|e| {
            VoirsError::internal("ProgressTracker", format!("Failed to update progress: {e}"))
        })?;

        Ok(())
    }

    pub async fn increment(&self, delta: u64) -> VoirsResult<()> {
        let current = self.receiver.borrow().current;
        self.update(current + delta, None).await
    }

    pub async fn set_message(&self, message: String) -> VoirsResult<()> {
        let current = self.receiver.borrow().current;
        self.update(current, Some(message)).await
    }

    pub async fn complete(&self) -> VoirsResult<()> {
        let total = self.receiver.borrow().total;
        self.update(total, Some("Completed".to_string())).await
    }

    pub fn subscribe(&self) -> watch::Receiver<ProgressInfo> {
        self.receiver.clone()
    }
}

pub struct TimeoutFuture<F> {
    future: F,
    timeout: Pin<Box<Sleep>>,
}

impl<F> TimeoutFuture<F> {
    pub fn new(future: F, timeout: Duration) -> Self {
        Self {
            future,
            timeout: Box::pin(tokio::time::sleep(timeout)),
        }
    }
}

impl<F> Future for TimeoutFuture<F>
where
    F: Future + Unpin,
{
    type Output = Result<F::Output, VoirsError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        if let Poll::Ready(()) = Pin::new(&mut this.timeout).poll(cx) {
            return Poll::Ready(Err(VoirsError::TimeoutError {
                operation: "Operation timed out".to_string(),
                duration: std::time::Duration::from_secs(0),
                expected_duration: None,
            }));
        }

        let future = unsafe { Pin::new_unchecked(&mut this.future) };
        match future.poll(cx) {
            Poll::Ready(result) => Poll::Ready(Ok(result)),
            Poll::Pending => Poll::Pending,
        }
    }
}

pub struct CancellableFuture<F> {
    future: F,
    token: CancellationToken,
}

impl<F> CancellableFuture<F> {
    pub fn new(future: F, token: CancellationToken) -> Self {
        Self { future, token }
    }
}

impl<F> Future for CancellableFuture<F>
where
    F: Future,
{
    type Output = Result<F::Output, VoirsError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if futures::executor::block_on(self.token.is_cancelled()) {
            return Poll::Ready(Err(VoirsError::cancelled("Operation was cancelled")));
        }

        let future = unsafe { self.as_mut().map_unchecked_mut(|s| &mut s.future) };
        match future.poll(cx) {
            Poll::Ready(result) => Poll::Ready(Ok(result)),
            Poll::Pending => Poll::Pending,
        }
    }
}

pub struct BufferedStream<T> {
    buffer: VecDeque<T>,
    capacity: usize,
    is_closed: bool,
}

impl<T> BufferedStream<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            is_closed: false,
        }
    }

    pub fn push(&mut self, item: T) -> Result<(), T> {
        if self.is_closed {
            return Err(item);
        }

        if self.buffer.len() >= self.capacity {
            return Err(item);
        }

        self.buffer.push_back(item);
        Ok(())
    }

    pub fn pop(&mut self) -> Option<T> {
        self.buffer.pop_front()
    }

    pub fn close(&mut self) {
        self.is_closed = true;
    }

    pub fn is_closed(&self) -> bool {
        self.is_closed
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }
}

impl<T: Unpin> Stream for BufferedStream<T> {
    type Item = T;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if let Some(item) = this.buffer.pop_front() {
            Poll::Ready(Some(item))
        } else if this.is_closed {
            Poll::Ready(None)
        } else {
            Poll::Pending
        }
    }
}

pub struct RetryFuture<F, Fut> {
    factory: F,
    current_future: Option<Fut>,
    attempts: usize,
    max_attempts: usize,
    delay: Duration,
    backoff_factor: f64,
}

impl<F, Fut> RetryFuture<F, Fut>
where
    F: FnMut() -> Fut,
    Fut: Future,
{
    pub fn new(
        factory: F,
        max_attempts: usize,
        initial_delay: Duration,
        backoff_factor: f64,
    ) -> Self {
        Self {
            factory,
            current_future: None,
            attempts: 0,
            max_attempts,
            delay: initial_delay,
            backoff_factor,
        }
    }
}

impl<F, Fut, T> Future for RetryFuture<F, Fut>
where
    F: FnMut() -> Fut + Unpin,
    Fut: Future<Output = VoirsResult<T>> + Unpin,
{
    type Output = VoirsResult<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        loop {
            if this.current_future.is_none() {
                if this.attempts >= this.max_attempts {
                    return Poll::Ready(Err(VoirsError::InternalError {
                        component: "retry_future".to_string(),
                        message: "Max retry attempts exceeded".to_string(),
                    }));
                }

                this.current_future = Some((this.factory)());
                this.attempts += 1;
            }

            if let Some(future) = &mut this.current_future {
                let future = unsafe { Pin::new_unchecked(future) };
                match future.poll(cx) {
                    Poll::Ready(Ok(result)) => return Poll::Ready(Ok(result)),
                    Poll::Ready(Err(_)) => {
                        this.current_future = None;

                        if this.attempts < this.max_attempts {
                            let delay = this.delay;
                            this.delay = Duration::from_secs_f64(
                                this.delay.as_secs_f64() * this.backoff_factor,
                            );

                            let sleep_future = tokio::time::sleep(delay);
                            tokio::pin!(sleep_future);

                            if sleep_future.poll(cx).is_pending() {
                                return Poll::Pending;
                            }
                        }

                        continue;
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }
        }
    }
}

pub fn timeout<F>(future: F, duration: Duration) -> TimeoutFuture<F>
where
    F: Future,
{
    TimeoutFuture::new(future, duration)
}

pub fn cancellable<F>(future: F, token: CancellationToken) -> CancellableFuture<F>
where
    F: Future,
{
    CancellableFuture::new(future, token)
}

pub fn retry<F, Fut, T>(
    factory: F,
    max_attempts: usize,
    initial_delay: Duration,
    backoff_factor: f64,
) -> RetryFuture<F, Fut>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = VoirsResult<T>>,
{
    RetryFuture::new(factory, max_attempts, initial_delay, backoff_factor)
}

pub async fn race<F1, F2>(future1: F1, future2: F2) -> Result<F1::Output, F2::Output>
where
    F1: Future,
    F2: Future,
{
    tokio::select! {
        result1 = future1 => Ok(result1),
        result2 = future2 => Err(result2),
    }
}

pub async fn all<T, F>(futures: Vec<F>) -> VoirsResult<Vec<T>>
where
    F: Future<Output = VoirsResult<T>>,
{
    let results = futures::future::join_all(futures).await;
    let mut output = Vec::new();

    for result in results {
        output.push(result?);
    }

    Ok(output)
}

pub async fn any<T, F>(futures: Vec<F>) -> VoirsResult<T>
where
    F: Future<Output = VoirsResult<T>> + Unpin,
{
    if futures.is_empty() {
        return Err(VoirsError::InternalError {
            component: "async_primitives".to_string(),
            message: "No futures provided".to_string(),
        });
    }

    let (result, _, _) = futures::future::select_all(futures).await;
    result
}
