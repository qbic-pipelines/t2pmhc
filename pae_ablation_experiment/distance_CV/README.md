This project was executed to retrain t2pmhc with different distance thresholds for the edge generation.
This includes the gat and gcn model, for both models we tested AUC using 5 fold cross on the training dataset. The examined distances are 6,8,10,12 and 14 A. 
We report AUC including std for the 5fold CV for every distance. Additionally we report max memory requirements and run time for the graph generation and training process.
The original scripts used to submit the jobs as well as any custom changes to the t2pmhc package required to run the experiment are attached.
Also all the generated graphs as well as the trained models and the splits used are provided.