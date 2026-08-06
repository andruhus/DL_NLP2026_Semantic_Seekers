# Documentation sst classification
### Structure of the Documentation
- every headline is an idea
- unter the headline there is the explanation behind it
- Ideas come in chronological order
## ROPE
### Starting point
- As discussed in the lecture, Rope can help to boost performance due to more efficient positional encoding
- But after short researching and thinking it turns out, that this approach would not be meaningful, since bert works with absolute positions not with relative
## Classification Layer
### Starting point
- There is huge overfitting happening in the base form of (vgl log file)
- This might be avoided by a more complex Classifier at the End of the Model
- I think this because I guess the Model has more options to train more complex relationships.
- So the actual classification based on the embeddings can be "nonlinear"
- A good embedding is not enough, when there is not enough potential in the classifier to untangle the complexity.
### Result (complex_classifier-10-1e-05-sst.pt_2026-07-30_18-29-45.log)
- The accuracy for training data actually decreased, that ist kind of surprising. I guess with more epochs learned and better learning rate scheduler this might be a different story
- The dev performance on the other hand increased slightly (0.019), which was expected. Further gains are expected for more epochs and better learning rate management 
## Learning rate scheduler
### Starting point
- As discussed in the section above, learning rate management might be the cause of weaker performance of the more complex classifier model
- The test for 10 epochs might not be the most reasonable, for fast and comparable result this will still be the starting point
- Linear Layers have a lot of weights, so making them more complex makes also the loss landscape more complex
- A learning rate scheduler might help to find a better optimum (first overshoot and then get finer when it gets better)
