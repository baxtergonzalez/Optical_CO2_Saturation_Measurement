import Detect_Sample

new_sample = Detect_Sample.Sample(
    'Sample_10.jpeg',
     blur = 7,
      method = 'Median',
       lower_bound = 120,
        upper_bound = 250,
         central_percentage = .1)
# new_sample.edge_detect(new_sample.color_mask(new_sample.image))

new_sample.show()