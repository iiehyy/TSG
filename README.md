TSG
Enhanced Dynamics of IP Allocation: Fine-Grained IP Geolocation via Temporal-Spatial Correlation**  
IEEE Transactions on Networking*

Directory Structure

- data: Stores all datasets, including raw data, augmented data, geographical graphs, training graph data, and test graph data.

- model: Stores model parameters.

- lib
  - LNGeoAugment: Generates augmented data for training and testing.
  - GeoHashMap: Creates geographical graphs based on augmented data.
  - GeoGraph: Generates graph data based on geographical graphs and augmented data.
  - TSG: The core model framework.
  - Dataloader: Manages data loading during training.
  - Test_Dataloader: Manages data loading during testing.
 
    
- train_TSG: Multi-process training for the model.
- test: Model testing.
- train_create_data**: Generates training data.
- test_create_data**: Generates test data.

