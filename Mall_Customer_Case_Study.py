import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def main():

    print("=========== Step 1 : Load the Data ==============\n")

    df = pd.read_csv("Mall_Customers.csv")

    print(df.head())

    print("\n")

    print("=========== Data Loaded Succesfully =============\n")

    print("Missing Values\n")

    print(df.isnull().sum())

    print("=========== Step 2 : Feature selection ==========\n")

    X = df[["AnnualIncome", "SpendingScore"]]

    print("========== Selected Features ==============\n")
    print(X.head())

    print("\n")

    print("========= Step 3 : Scalled the data ============= \n")

    Scalar = StandardScaler()

    X_scaled = Scalar.fit_transform(X)

    print("========== Scalled data ========= \n")

    print(X_scaled[:5])

    print("\n")

    print("========= Step 4 : Elbow method ================ \n")

    WCSS = []      # WCSS = Within clustor sum of squares

    for k in range(1, 11):

        model = KMeans(
            n_clusters = k,
            random_state = 42,
            n_init = 10
        )

        model.fit(X_scaled)

        WCSS.append(model.inertia_)

    print("Values of WCSS")

    for i in range(len(WCSS)):
        print(f"{i + 1} : {WCSS[i]}")

    print("\n")

    print("======== Step 5 : Data Visualization=============\n")

    plt.plot(range(1, 11), WCSS, marker = "o")
    plt.xlabel("Number of Clusters : K ")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.show()

    print("\n")

    print("======== step 6 : Final Model =================== \n")

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(X_scaled)

    df["Cluster"] = clusters

    print("=========== Dataset with Clusters : ============ \n")
    print(df.head(20))



if __name__ == "__main__":
    main()