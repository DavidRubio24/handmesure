# Handmesure

Utilities to locate points of interest in images of scanned hands to mesure the hand.


This project aims to detect points in two different poses: opened and closed.
Each pose has its own points to locate, 23 for the opened hands and 15 for the closed ones.

As of right now (2022/6/15) the strategy is as follows:
1. Detect the hand: make sure that there's a hand in the image, locate it.
2. Zoom in the hand.
3. Estimate the location of the points of interest.
4. Show a human the points, so she can modify them if they are wrong.

Steps 1 is done with MediaPipe Hands.
Since we don't have a big enough dataset for step 3 we used the architecture and the
weights of the MediaPipe Hands solution, and we only changed the last layer (see model.py).
Step 1 is done flawlessly this way, but step 3 doesn't achive eonugh accuracy (1mm), this maybe do to the small size of
our dataset or the resizing of the image to 224x224 to feed the Neural Network.
Neither data augmentation nor changing the architecture to accept 448x448 images has improved the results enough (the latter has worsened them).