//! R Statistical Analysis Integration Example
//!
//! This example demonstrates how to use the R statistical analysis integration
//! for advanced statistical testing and modeling.
//!
//! Run with: cargo run --example r_statistical_analysis --features r-integration

use std::error::Error;
use voirs_evaluation::r_integration::{utils, RSession};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("VoiRS Evaluation - R Statistical Analysis Integration Example");
    println!("============================================================");

    // Check if R is available
    if !utils::is_r_available() {
        println!("❌ R is not available on this system.");
        println!("Please install R from https://www.r-project.org/");
        return Ok(());
    }

    // Get R version
    match utils::get_r_version().await {
        Ok(version) => println!(
            "✅ R Version: {}",
            version.lines().next().unwrap_or("Unknown")
        ),
        Err(e) => println!("⚠️  Could not get R version: {}", e),
    }

    // Create R session
    println!("\n🚀 Creating R session...");
    let mut r_session = RSession::new().await?;
    println!("✅ R session created successfully");

    // Example 1: Basic T-test
    println!("\n📊 Example 1: One-sample T-test");
    println!("Testing if mean of [1,2,3,4,5] equals 3.0");

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let t_test_result = r_session.t_test(&data, Some(3.0)).await?;

    println!("T-test Results:");
    println!("  Statistic: {:.4}", t_test_result.statistic);
    println!("  P-value: {:.4}", t_test_result.p_value);
    println!("  Method: {}", t_test_result.method);

    if let Some((lower, upper)) = t_test_result.confidence_interval {
        println!("  95% CI: [{:.4}, {:.4}]", lower, upper);
    }

    if t_test_result.p_value < 0.05 {
        println!("  Result: ❌ Reject null hypothesis (mean ≠ 3.0)");
    } else {
        println!("  Result: ✅ Fail to reject null hypothesis (mean = 3.0)");
    }

    // Example 2: Two-sample t-test using Wilcoxon test
    println!("\n📊 Example 2: Wilcoxon Test (Two independent samples)");

    let group1 = vec![1.2, 1.8, 2.1, 2.3, 2.7];
    let group2 = vec![2.8, 3.1, 3.4, 3.9, 4.2];

    println!("Group 1: {:?}", group1);
    println!("Group 2: {:?}", group2);

    let wilcox_result = r_session.wilcox_test(&group1, Some(&group2)).await?;

    println!("Wilcoxon Test Results:");
    println!("  Statistic: {:.4}", wilcox_result.statistic);
    println!("  P-value: {:.4}", wilcox_result.p_value);
    println!("  Method: {}", wilcox_result.method);

    if wilcox_result.p_value < 0.05 {
        println!("  Result: ✅ Significant difference between groups");
    } else {
        println!("  Result: ❌ No significant difference between groups");
    }

    // Example 3: Linear Regression
    println!("\n📊 Example 3: Linear Regression");

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1];

    println!("X values: {:?}", x);
    println!("Y values: {:?}", y);

    let regression_result = r_session.linear_regression(&x, &y).await?;

    println!("Linear Regression Results:");
    if let Some(intercept) = regression_result.coefficients.get("(Intercept)") {
        println!("  Intercept: {:.4}", intercept);
    }
    if let Some(slope) = regression_result.coefficients.get("x") {
        println!("  Slope: {:.4}", slope);
    }
    println!("  R-squared: {:.4}", regression_result.r_squared);
    println!(
        "  Adjusted R-squared: {:.4}",
        regression_result.adjusted_r_squared
    );
    println!("  F-statistic: {:.4}", regression_result.f_statistic);
    println!("  F p-value: {:.4}", regression_result.f_p_value);

    if regression_result.f_p_value < 0.05 {
        println!("  Result: ✅ Regression model is statistically significant");
    } else {
        println!("  Result: ❌ Regression model is not statistically significant");
    }

    // Example 4: Correlation Analysis
    println!("\n📊 Example 4: Correlation Analysis");

    let var1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let var2 = vec![2.0, 4.1, 5.8, 8.2, 9.9, 12.1, 14.0];

    println!("Variable 1: {:?}", var1);
    println!("Variable 2: {:?}", var2);

    // Pearson correlation
    let pearson_result = r_session.correlation(&var1, &var2, "pearson").await?;
    println!("Pearson Correlation:");
    if let Some(estimate) = pearson_result.parameters.get("estimate") {
        println!("  Correlation: {:.4}", estimate);
    }
    println!("  P-value: {:.4}", pearson_result.p_value);

    // Spearman correlation
    let spearman_result = r_session.correlation(&var1, &var2, "spearman").await?;
    println!("Spearman Correlation:");
    if let Some(estimate) = spearman_result.parameters.get("estimate") {
        println!("  Correlation: {:.4}", estimate);
    }
    println!("  P-value: {:.4}", spearman_result.p_value);

    // Example 5: ANOVA
    println!("\n📊 Example 5: One-way ANOVA");

    let group_a = vec![2.3, 2.5, 2.7, 2.9, 3.1];
    let group_b = vec![3.2, 3.4, 3.6, 3.8, 4.0];
    let group_c = vec![4.1, 4.3, 4.5, 4.7, 4.9];

    println!("Group A: {:?}", group_a);
    println!("Group B: {:?}", group_b);
    println!("Group C: {:?}", group_c);

    let groups = vec![group_a, group_b, group_c];
    let anova_result = r_session.anova(&groups).await?;

    println!("ANOVA Results:");
    for (i, source) in anova_result.sources.iter().enumerate() {
        if i < anova_result.f_statistics.len() && !anova_result.f_statistics[i].is_nan() {
            println!(
                "  {}: F({},{}) = {:.4}, p = {:.4}",
                source,
                anova_result.df[i],
                anova_result.df.get(i + 1).unwrap_or(&0),
                anova_result.f_statistics[i],
                anova_result.p_values[i]
            );
        }
    }

    if !anova_result.p_values.is_empty() && anova_result.p_values[0] < 0.05 {
        println!("  Result: ✅ Significant difference between groups");
    } else {
        println!("  Result: ❌ No significant difference between groups");
    }

    // Example 6: Custom R Script
    println!("\n📊 Example 6: Custom R Script Execution");

    let custom_script = r#"
        # Generate some sample data
        set.seed(123)
        x <- rnorm(50, mean = 100, sd = 15)
        
        # Calculate descriptive statistics
        cat("Sample Size:", length(x), "\n")
        cat("Mean:", mean(x), "\n")
        cat("Median:", median(x), "\n")
        cat("Standard Deviation:", sd(x), "\n")
        cat("Min:", min(x), "\n")
        cat("Max:", max(x), "\n")
        
        # Perform normality test
        shapiro_result <- shapiro.test(x)
        cat("Shapiro-Wilk p-value:", shapiro_result$p.value, "\n")
    "#;

    println!("Executing custom R script for descriptive statistics...");
    let custom_output = r_session.execute_script(custom_script).await?;
    println!("Custom Script Output:");
    for line in custom_output.lines() {
        if !line.trim().is_empty() {
            println!("  {}", line);
        }
    }

    println!("\n✅ R Statistical Analysis Integration Demo Complete!");
    println!("\n💡 Tips:");
    println!("  - Install R packages as needed: install.packages('package_name')");
    println!("  - Use the install_package() method for automatic package management");
    println!("  - Combine with VoiRS evaluation metrics for advanced analysis");
    println!("  - Leverage R's extensive statistical and visualization libraries");

    Ok(())
}

#[cfg(not(feature = "r-integration"))]
fn main() {
    println!("This example requires the 'r-integration' feature.");
    println!("Run with: cargo run --example r_statistical_analysis --features r-integration");
}
