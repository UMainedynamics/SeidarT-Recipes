Common Offset
-------------

This script simulates a typical GPR survey that uses a common offset between the source and single receiver. SeidarT computes a model for each source-receiver pair and extracts the time series prior to deleting all of the model outputs. The source-receiver pairs are in the order that they are given in the CSV files containing their coordinates. There are 3 files that need to be declared. These are the project file, receiver coordinate file, and source coordinate file. 