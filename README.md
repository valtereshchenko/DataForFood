# DataForFood

Nowadays it is quite complicated to make healthy food choices. There are various apps out there suggesting you whether one product is healthier than another. 
In this case, I have decided to explore the score most of the 'healthy' apps are using - Nutriscore -
that suggest on the scale from A to E how healthy of unhealthy the certian product is. 

I have explored OpenFoodFacts database of products & scores those have, to then build a model that wouls calculate the Nutriscore 
for the product that are not yet in the database.

I have used Tensorflow to build a model that would detect the nutritional label on the products' pictures. 

After that, with the help of Google Vision API I was able to extract the text from the previously detected nutritional labels. 

Finally I have build the SVM regression model that predicts the nutriscore for the newly added products (based on the data from the OpenFoodFacts),
followed by the Random Forest Classifier that predicts then the corresponding label of the product (from A to E).


