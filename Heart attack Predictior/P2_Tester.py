import numpy as np

def data_colleter():
    Age=int(input("Enter you're age : "))
    Gender=int(input("Gender (1-M or 0-F) : "))
    Heart_rate=int(input("Enter Heart rate : "))
    systolic_blood_pressure =int(input("Enter systolic blood pressure : "))
    Diastolic_blood_pressure =int(input("Enter Diastolic blood pressure : ")) 
    Blood_sugar = int(input("Enter Blood sugar: "))
    Creatine_Kinase_MB = float(input("Enter Creatine Kinase MB : ")) 
    Troponin=float(input("Enter Troponin level : "))
    return [Age,Gender,Heart_rate,systolic_blood_pressure,Diastolic_blood_pressure,Blood_sugar,Creatine_Kinase_MB,Troponin]

def sigmoid(z):
    z = np.clip(z, -500, 500)   
    return 1 / (1 + np.exp(-z))

def test(lis):
    lis = np.array(lis, dtype=float).flatten()
    X_test = lis.reshape(1, -1)

    W1=np.array([[ 0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337,
        -0.23413696,  1.57921282,  0.76743473],
       [-0.46947439,  0.54256004, -0.46341769, -0.46572975,  0.24196227,
        -1.91328024, -1.72491783, -0.56228753],
       [-1.01283112,  0.31424733, -0.90802408, -1.4123037 ,  1.46564877,
        -0.2257763 ,  0.0675282 , -1.42474819],
       [-0.54438272,  0.11092259, -1.15099358,  0.37569802, -0.60063869,
        -0.29169375, -0.60170661,  1.85227818],
       [-0.01349722, -1.05771093,  0.82254491, -1.22084365,  0.2088636 ,
        -1.95967012, -1.32818605,  0.19686124]])
    W2= np.array([[ 0.59781036  ,0.17136828 ,-0.11564828 ,-0.3011037  , 2.34756522]])
    b1= np.array([[0.00000000e+00, 3.36689104e-77, 2.38115427e-47, 1.17275028e-29,1.09542436e-59]])
    b2= np.array([[-0.14065622]])


    Z1 = np.dot(X_test, W1.T) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2.T) + b2
    A2 = sigmoid(Z2)

    pred = 1 if A2[0][0] >= 0.5 else 0
    return pred  

print("Possitive" if test(data_colleter())==1 else "Negative")


58, 1, 116, 95, 99, 120, 8.65, 0.231

