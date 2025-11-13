I like the proposed changes. When you mention threads what are you meaning exactly? In the near future I plan on adding 
and utilizing both TensorRT and CUDA for inference optimizations, is it related to the threads you mentioned?
the idea of running on GPU is great. just for you to have noticed on this I made a change and have the nvidia drivers setup for default use now in order for future implementation of Tensrflow/CUDA/TensorRT etc.
My device has the itel core ultra 9 but most importantly it has the nvidia GEOFORCE RTX series 5060 8GB GDDR7
I dont know if the use of this NVIDIA driver is what you had in mind when mentioning Yolo on GPU.
I already made the change on the analyze every 10 frames instead of every 1
If I undertsand correctly the step 4) implementation of a small cache, is to store the last age/gender of the detected face 
and since the age/gender does not change that often, we can use the cached value for the next frames until a certain number of frames is reached.
I didn't quite understand the step 5).

I also think that if we where to implement a pipeline where we first detect faces track them and assign an ID to each face there os no need to constantly run the age/gender inference over every face over and over
so ,aybe when assigning an ID to a new detected face we can also end up only running the age/gender one time to that idientified face and store the result in a dictionary with the ID as key and the age/gender as value.
Then for the next frames we just retrieve that info. 
    
     
