# similarity-embeddings

Codebase for similarity-based vision embeddings.


# Instructions for evaluating students vs teachers

- Train a teacher model using the --save-each-epoch flag in 'run_training' 

This flag doesn't actually save each epoch but saves a version of the teacher model each time it improves on the test set. I recommend then moving the directory containing 
this teacher model to something nicely named.

- Train students on each version of the teacher model

Use the 'train_many_models' file. Provide the directory to the teachers and a few extra training options. I recommend you start with an empty log directory and when all students are trained move
the students to some nicely named directory. 

You probably want to provide more than 1 GPU for training at this stage. The multi-processing is kinda hacky but does work well for this application.

- Add a linear classifier to the students

This is neccessary if you trained the students via similarity. The scripts will rename your students based on information from the corresponding log file to match the epoch. 
For this reason include the '--rename' flag. 

- Evaluate your teachers and students

This can be done using the 'evauluate_many_models' file




