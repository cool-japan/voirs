use voirs_feedback::training::InteractiveTrainer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let trainer = InteractiveTrainer::new().await?;
    let library = trainer.exercise_library.read().unwrap();
    let total_exercises = library.exercises.len();
    
    println\!("Total exercises in library: {}", total_exercises);
    
    // Count by difficulty level
    let beginner = library.exercises.iter().filter( < /dev/null | e| e.difficulty <= 0.3).count();
    let intermediate = library.exercises.iter().filter(|e| e.difficulty > 0.3 && e.difficulty <= 0.6).count();
    let advanced = library.exercises.iter().filter(|e| e.difficulty > 0.6).count();
    
    println!("Beginner exercises (0.1-0.3): {}", beginner);
    println!("Intermediate exercises (0.4-0.6): {}", intermediate);
    println!("Advanced exercises (0.7-1.0): {}", advanced);
    
    Ok(())
}
