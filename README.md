# Best Checkpoint Copier for Pytorch
A class that copies the best checkpoints for pytorch
## Usage
```
  best_saver = BestCheckpointCopier(
    name='best', # directory within model directory to copy checkpoints to
    checkpoints_to_keep=10, # number of checkpoints to keep
    compare_fn=lambda x,y: x.score < y.score, # comparison function used to determine "best" checkpoint (x is the current checkpoint; y is the previously copied checkpoint with the highest/worst score)
    sort_key_fn=lambda x: x.score, # key to sort on when discarding excess checkpoints
    sort_reverse=False) # sort order when discarding excess checkpoints
  
  .
  .
  .
  for epoch in range(1, opt.nEpochs + 1):
        train(epoch) #training
        score = test() #testing with returning score
        save_iter_checkpoint(epoch) #save model for each epoch
        BestCheckpointCopier(score, checkpoint_path)
  .
  .
  .
```
## Reference
https://github.com/bluecamel/best_checkpoint_copier.git
