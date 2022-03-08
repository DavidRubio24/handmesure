# Handmesure

Utilities to locate points of interest in images of scanned hands to mesure the hand.


This project aims to detect points in two different poses: opened and closed.
Each pose has its own points to locate, 23 for the opened hands and 15 for the closed ones.

As of right (2022/3/8) now the strategy is as follows:
1. Detect the hand: make sure that there's a hand in the image, locate it and determine if it's a right hand or left hand.
2. Crop the hand from the image. Flip it if it is a left hand.
3. Estimate the location of the points of interest.
4. Refine the location of the points of interest.
5. Show a human the points in decreasing order of confidence. Let her modify them if they are wrong.

Steps 1 and 3 are done with a neural network. Since we don't have a big enough dataset we used the architecture and the weights of the MediaPipe Hand solution and we only changed the last layer (see model.py).
Step 1 is done flawlessly this way, but step 3 does't achive eonugh accuracy (1mm), this may be do to the small size of out dataset or the resizing of the image to 224x224 to feed the Neural Network. Neither data augmentation nor changing the architecture to accept 448x448 images has improved the results enough (the later has worsened them). Our next attempt will try to use classic CV to enhance the NN estimation.