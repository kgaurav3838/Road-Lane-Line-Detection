# Road Lane-Line Detection

Road lane-line detection is a primary function for an autonomous car and has been applied
in various smart vehicle systems, but it is challenging for computer vision. It helps build a
decision-making and path planning algorithm and more like steering control, accelerating
action, breaking, relative vehicle trajectory motions, and much more. This algorithm
provides an accurate and reliable fit to road lane-lines, and with it, we can easily estimate
image sequence alignment such as drift, shift, rotation, etc. where we can adjust or align the frame
according to our requirement. We have created a chart explaining the steps involved in the
real-time road lane-line detection and image sequence alignment. The detailed steps
involved in our road lane-line detection are given as follows:

**Load the target image:** We can easily capture and load the test images one by one in which we have to detect lane-lines using OpenCV inbuilt features.

**Selection of suitable Color:** This is the fundamental part that helps us locate the road dashed lane-line so that we can extract it in our image views and aim for it. Pictures are basically in RGB format. To underline our lane line, we have to choose different color spaces like HSL, HSV.  We specifically chose the HSL color space to opt-out of the dashed lane lines. Once this color selection is applied to HSL images, it will suppress everything except for the white (in our case) lane lines.
    
**Smoothening of the images:** Before proceeding to locate the edges in the image, we have to do some smoothing of our target images. As we know, edge detection methods include measurement of intensity gradients, so we need to convert our images to grayscale to detect edges in our image because it is faster than handling an RGB colored image. After that, we have to apply some filters to smooth the edges and remove noises as noise can create false edges. We have used Gaussian filters to achieve our objectives.
    
**Canny Edge Detection:**  For edge detection in the image, the Canny edge detection technique is beneficial for detecting road-dashed lane-lines successfully. It measures gradient in several directions and tracks the edges of our blurred picture with a large intensity shift \cite{canny}.
  
**Selecting a suitable region of interest:** The field facing the camera is of concern, where there are lane lines. Therefore, we apply a mask to this area, and everything else will be suppressed.

**Hough Transform:**   In image processing, line detection is an algorithm that takes a collection of edge points and finds all the lines on which these edge points lie. The Hough transform is a procedure used to separate components of a particular shape in a picture. The most popular line detector is the Hough transform techniques. The notion of Hough transformation is for each edge point in the edge map to be converted into the line conceivable across that point.

**Lane lines evaluation, averaging, and extrapolating:** For each side, we have several lines identified. Both these lines must be combined, and a single line drawn for each line must be drawn. Lane lines can either be extended or extrapolated to the longest path.

**Alignment of Image sequence:** Once the lane-line is detected and extrapolated, we can easily find the intercept and slope of the line of the first frame. We can further apply a linear regression technique to find out the average slope and intercept. We are going to relate this slope and intercept with the subsequent frames. Upon successfully estimating the difference in slope and intercept, we can easily align and warp the image sequences. 
    
**Click to open to View the real-time Road Lane-Line Detection results:-**
Link:- https://iitk-my.sharepoint.com/:v:/g/personal/kkgaurav_iitk_ac_in/ETyuArlvuB9Kn1Jad-5_-IQBD8C2f1d0E9y6_jXLWGbi1w?e=Bwx7wz
