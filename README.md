# Optical_CO2_Saturation_Measurement
Take an image of an indicated concrete sample, locate sample within image, extract contour to represent sample, extract contour to represent non-CO2 saturated areas, then return several useful metrics

Purpose:
Many experiments in concrete labs use phenolphthalein as an indicator. The region of a sample that is colored is more basic, in my case because it has not absorbed CO2. 

It is useful to know several metrics about this colored region in relation to the overall sample;
-What is the maximum depth that CO2 has penetrated into the sample? This can inform operators how far to place their rebar within a beam to prevent corrosion and is very difficult to determine by hand.
-How much of the sample has absorbed CO2? This can give indication to the strength and chemical resistance of the material. Additionally, for the non-uniform shapes of the indicated regions the area can be nearly impossible to calculate by hand.
-What is the average penetration of CO2 in the sample? This can also educate rebar placement choices, and can only be roughly approximated by hand measurement techniques.

All of these problems indicate a need for a better option of measurement. I decided on an optical approach, because of the visual nature of the indicator solution.

Overview:

Pre-processing
  -An image is loaded in
  -A series of pre processing steps occur to make computer vision techniques easier and more effective
  -An edge finding algorithm is applied, followed by a contour finding algorithm
Filtering and locating
  -The returned contours are filtered by their location within the image, shape, and area in pixels in order to determine which contour is the sample
  -Once the sample contour has been located, a color mask is applied to isolate the indicated region within the sample
  -Edge finding and contour finding algorithms are employed to determine a contour for the indicated region
Measurement
  -The Area of the contours are calculated using cv2 functions, yielding percent area indicated
  -The maximum and average penetration of CO2 is determined using a ray casting algorithm
    -First the centroid of the contour is calculated
    -The contours are translated so that the centroid is at the origin
    -The coordinates are converted from cartesian to polar
    -The contours are sorted by their theta component
    -Linear interpolation between points on contour with less points to generate the missing points (The contours are not always generated with the same number of points)
    -The radius component of the inner contour is subtracted from the radius component of the outer contour that shares a theta value
    -This yields a curve representing the penetration of CO2
    -This curve can be processed to determine average penetration as well as max and min penetration


