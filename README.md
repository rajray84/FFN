How to run
-----------
1. Enter into FFN directory and create an empty folder to save .onnx file and property file 

       cd FFN
       mkdir uploads

2. 
    a. FFN : Run in local server using a docker image
     
       sudo docker build . -t ffn_image
       sudo docker run -t ffn_image

    b. FFN : Run in local server without docker image
    
       pip install Flask
       cd FFN
       python app.py
    

To check : 
--------
   http://127.0.0.1:5000/verify
       
  
   
