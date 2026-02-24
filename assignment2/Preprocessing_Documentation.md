# NYC Taxi Trip Duration - Preprocessing Documentation

## Feature Selection and Engineering
For predicting trip duration, we extracted several key features from the raw dataset:

1.  **Datetime Extraction**: The `pickup_datetime` was converted into:
    *   **Month**: Captures seasonal variations.
    *   **Day of Week**: Captures differences between weekday and weekend traffic patterns.
    *   **Hour**: Captures daily peak hours (rush hour vs. night).
2.  **Geospatial Data**:
    *   `pickup_longitude`, `pickup_latitude`
    *   `dropoff_longitude`, `dropoff_latitude`
3.  **Metadata**:
    *   `vendor_id`: Taxi provider.
    *   `passenger_count`: Number of passengers.

## Transformations
*   **Target Transformation (Log-Scaling)**: Trip durations often have a long-tailed distribution. To normalize the target and reduce the impact of outliers, we applied a log transformation: `y' = log(1 + trip_duration)`.
*   **Feature Normalization**: All features were standardized (Z-score normalization) using the mean and standard deviation of the training set. This ensures that features like longitude/latitude and hours are on a comparable scale, preventing the model from becoming biased toward larger numeric values.

## Model Experimentation
Three different architectures were tested to find the optimal balance between complexity and generalization:
1.  **Config 1**: Hidden layers [64, 32], Learning Rate 0.01.
2.  **Config 2**: Hidden layers [128, 64], Learning Rate 0.005.
3.  **Config 3**: Hidden layers [64, 64, 32], Learning Rate 0.01.

Plots for the training and validation loss of each configuration are generated as `Config_X_plot.png`.
