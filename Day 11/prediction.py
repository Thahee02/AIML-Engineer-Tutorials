import joblib
import pandas as pd

model = joblib.load("house_price_model.pkl")

area = float(input("\nEnter Area (in sq ft): "))
bedrooms = int(input("Enter Number of Bedrooms: "))
bathrooms = int(input("Enter Number of Bathrooms: "))
age = int(input("Enter Age of the House (in years): "))
distance = float(input("Enter Distance from City Center (in km): "))

new_house = pd.DataFrame({
    "Area": [area],
    "Bedrooms": [bedrooms],
    "Bathrooms": [bathrooms],
    "Age": [age],
    "Distance": [distance]
})

predictions = model.predict(new_house)

print(
    "Predicted Price:", predictions[0]
)