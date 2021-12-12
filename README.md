## Grid detectors
help an object detection model detects multiple objects at different locations. This is done by dividing the input image into multiple grid cells (e.g. 3 x 3, 10 x 10) where each grid cell has its own simple object detection model. Each detector specializes in detecting and classifying objects that fall inside their respective grid cell.
## Default boxes 
allow an object detection model to detect objects with different shapes. These default boxes sets up more constraints that allow a bounding box predictor to focus on only detecting objects of particular shape. The shape of default box is represented by its width and height. Each default box has object detection model that specializes in predicting object with that particular shape.
## Feature maps 
correspond to a specific location/patch on the input image. The deeper a convolutional layer is in the CNN, the more descriptive the feature map gets. In the first few layers of a CNN, the feature map items produced by those layers correspond to small patches that are edges, lines or corners in the original image. Further down the CNN, the feature map items corresponds to bigger patches on the input image.Those big patches mights be part of an object or even full objects.
## Base work 
is a CNN network. This is because networks that performs well on image classification tasks have also proven to be good feature maps extractors for object detection models. When training an Object detection model, these base networks are usually pre-trained and their weights are kept unchanged during the training process.
## Convolutional Predictors
CNN are created to deal with image classification tasks produce its final output through a fully connected layer. However, the usage of fully connected layers presented two issues: first, it can lead to the lost of the spatial information given by the feature maps. Second, the number of parameters can get very large if there are a lot of feature maps. Convolutional Predictors can solve these issues. A small convolution filter ( e.g. size 1*1, 3*3, etc) is used to produce the predictions. We are not transforming a feature map from a size of width* height*channels into a 1D vector for a fully connected layers, spatial information is not lost and the convolutional predictor will just produce predictions of size 38*38*num_classes instead of 38*38*512=739328 parameters.
The SSD network structure
The core idea behind SSD network is to have a CNN that takes in an image as input and produce detections at different scales, shapes, and locations.
SSD300 SSD500 difference: input size 
## The SSD Loss function
SSD output has the shape of ( total_default_boxes, num_classes +1+4+8)
What included in each total_default_boxed item are:
1. Confidence scores for num_classes +1 background class
2. 4 bounding box properties: cx ( the x offset to the matched defaults box’s center), cy (the y offset to the matched default box’s center), the log scale transform of the width of the bounding box (w),  and the log scale transform of the height of the bounding box (h)
3. 4default values: the center x offset of the default box from the left of the image, the center y offset of the default box from the top of the image, the width of the default box, the height of the default box.
4. 4 variance values: the values used to encode/ decode bounding box data.
1, 2 are learnable by the ssd network and 3, 4 are constant.
SSD loss function combines together regression loss (L_loc) and the classification loss(L_conf). L_loc is the sum of smooth L1 loss across all bounding box properties( cx,cy,w,h) for matched  positive boxes. This means that it does not tale into account default boxes whose classes are the background class or default boxes that do not matched with any ground truth boxes. Through this, it will be rewarded for making bounding box predictions that have objects in them. 
L_conf sum the loss for matches positives default boxes and negative default boxes. Why negative boxes are also factored into classification loss is because we want to punish wrong predictions.
## Code:
1. 1configs. Create a config file to store all parameters
2. 2custom_layers. Construct DefaultBoxes and L2 Normalization Layer
DefaultBoxes layers: During the training stage the values from this layer are not needed. However, during inference stage, the values from this layer will be crucial for decoding bounding box predictions produced by the network.
L2 Normalization Layer: This layer is used to apply L2 Normalization with a learnable scale value. It will only be used on the conv4_3 feature maps layer. 
3. 3networks. Construct the ssd network:
  1)Constructs the base network, loads a pre-trained weights, and freeze the base network layers so that its weights will not changed during training.
  2)Constructs the SSD’s extra feature layers.
  3)Determines all the possible scales for default boxes.
  4)For each feature map: 
	a. apply 3 x 3 Convolutional Predictors to produce classification predictions of shape (w, h, num_default_boxes*(num_classes + 1)) and localization predictions of shape (w, h, num_default_boxes*4) where w and h are the width and height of the feature map respectively
  b.reshape those predictions to a shape of (w*h*num_default_boxes, num_classes+1) for classifications and (w*h*num_default_boxes, 4) for localization. 
  c. generate default boxes for that particular layer and reshape it from a shape of (w, h, num_default_boxes, 8) to a shape of (w*h*num_default_boxes, 8)
  5)Concatenate classification predictions from every feature maps together and apply a Softmax activation to get the final classification results.
  6)Concatenate localization predictions for every feature maps together
  7)Concatenate all default boxes for every feature maps layers together
  8)Concatenate all classifications, localizations, and default boxes together to produce a final output of shape (total_default_boxes, num_classes + 1 + 4 + 8)
 4. loss. SSD Loss function: smooth L1 loss + softmax loss
