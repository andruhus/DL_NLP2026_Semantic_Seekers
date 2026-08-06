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
- A learning rate scheduler might help to find a better optimum (first warmup at the end get finer when it gets better)
### Results
- It turns out that the scheduler further reduces overfitting (on top of the more complex classifier)
- The training accuracy goes further down. This does not have to be a drawback, since it might highlight more capacity for improvement for later epochs.
- Val performance on the other hand goes marginally up. The potential of the result lies not in the direct performance increase but rather in the potential for longer training with adapted initial learning rate.
## More complex pooling layer (here a paper is needed)
### starting point
- Currently the classification only depends on the CLS token
- Some words might be quite important, that might be neglected in the CLS token (WHY???)
- Lets try to add maxpooling and mean pooling
### result
- pooling actually only increased training accuracy marginally 
- val performance decreased marginally, so overfitting might be the problem again.
- WHY? -> model might lack complexity to learn the new relationships in a reasonable way? 
- possible fixing strategies:
  1. more layers in the classifier part? (we go down from 3*Bertembedding to 512 immediatelly + this approach worked fine in the first step es well)
  - reduces dev only marginally, but less overfitting -> good
  - with smaller learning rate might be better (vgl logfile 3.1)
  - -> it turns out that wether higher nor lower lr improves performance, it actually maked it really bad :(
  - compared to model 2 its still no improvement
  - based on literature not really successfull training curve its still worth the effort for deeper investigations
  2. only use mean pooling (Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks” (Reimers & Gurevych, 2019))
  3. generell paper gegen overfitting (How to Fine-Tune BERT for Text Classification?” (Sun et al., 2019))
  4. und hier nochmal genauer was CLS überhaupt soll (A Primer in BERTology: What we know about how BERT works” (Rogers et al., 2020))
  5. Layer norm unbedingt nach pooling (in forward)
