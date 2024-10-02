import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.tree import DecisionTreeRegressor

# Đọc dữ liệu và xử lý đầu vào chỉ thực hiện một lần
@st.cache_resource
def load_data_and_models():
    # Đọc dữ liệu
    cars = pd.read_csv('CarPrice_Assignment.csv')
    
    # Tách đặc trưng (features) và nhãn (target)
    X = cars.drop(['price', 'CarName', 'car_ID'], axis=1)
    y = cars['price']
    
    # One-hot encoding cho các cột dạng object
    X = pd.get_dummies(X, drop_first=True)
    
    # Chia dữ liệu thành train và test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Huấn luyện các mô hình
    lin_reg = LinearRegression().fit(X_train_scaled, y_train)
    ridge_reg = Ridge(alpha=5.0).fit(X_train_scaled, y_train)
    nn_reg = MLPRegressor(hidden_layer_sizes=(150, 75), max_iter=5000, learning_rate_init=0.01,
                          random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10).fit(X_train_scaled, y_train)
    
    stacking_reg = StackingRegressor(
        estimators=[('lr', lin_reg), ('ridge', ridge_reg), ('nn', nn_reg)],
        final_estimator=DecisionTreeRegressor()
    ).fit(X_train_scaled, y_train)
    
    return scaler, lin_reg, ridge_reg, nn_reg, stacking_reg, X.columns

# Chỉ tải và khởi tạo mô hình một lần
scaler, lin_reg, ridge_reg, nn_reg, stacking_reg, X_columns = load_data_and_models()

# Định nghĩa hàm dự đoán
def predict_price(model, data):
    input_data = pd.DataFrame([data])
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X_columns, fill_value=0)
    input_data_scaled = scaler.transform(input_data)
    
    # Kiểm tra dữ liệu đầu vào
    if input_data_scaled.shape[1] != len(X_columns):
        st.error("Dữ liệu đầu vào không hợp lệ!")
        return None
    
    return model.predict(input_data_scaled)[0]

# Giao diện người dùng
st.title("Car Price Prediction App")
st.write("Chọn mô hình để dự đoán giá xe.")

# Nút chọn mô hình
model_options = {
    'Linear Regression': lin_reg,
    'Ridge Regression': ridge_reg,
    'Neural Network': nn_reg,
    'Stacking Model': stacking_reg
}

selected_model_name = st.selectbox("Chọn mô hình:", list(model_options.keys()))
selected_model = model_options[selected_model_name]

# Nhập dữ liệu từ người dùng
wheelbase = st.number_input('Wheelbase', value=98.4)
carlength = st.number_input('Car Length', value=168.8)
carwidth = st.number_input('Car Width', value=64.1)
curbweight = st.number_input('Curb Weight', value=2548)
enginesize = st.number_input('Engine Size', value=130)
horsepower = st.number_input('Horsepower', value=111)
peakrpm = st.number_input('Peak RPM', value=5000)
citympg = st.number_input('City MPG', value=21)
highwaympg = st.number_input('Highway MPG', value=27)

new_data = {
    'wheelbase': wheelbase,
    'carlength': carlength,
    'carwidth': carwidth,
    'curbweight': curbweight,
    'enginesize': enginesize,
    'horsepower': horsepower,
    'peakrpm': peakrpm,
    'citympg': citympg,
    'highwaympg': highwaympg,
}

if st.button("Dự đoán giá"):
    predicted_price = predict_price(selected_model, new_data)
    if predicted_price is not None:
        st.write(f"Giá dự đoán: ${predicted_price:,.2f}")
