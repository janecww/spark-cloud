import numpy as np
import pandas as pd
# import pyspark

def predictRatings(data, target):
    data = pd.DataFrame(data, columns=["UserID", "ItemID", "Rating", "Timestamp"]).astype("int")
    matrix = data.pivot_table(values="Rating", columns="UserID", index=["ItemID"])
    friends = matrix.drop([target], axis=1)
    sim = friends.corrwith(matrix[target])
    result = matrix[target].mean() + (sim * (friends - friends.mean())).sum(axis=1) / sim.abs().sum()
    return result.sort_values(ascending=False)

def createDf(x):
    arr = np.array(x[1])
    df = pd.DataFrame(arr[:,1], index=arr[:,0]).sort_index()
    return (x[0], df)

def predictRatingsRDD(rdd, target):
    # rdd = sc.parallelize(data).groupByKey().mapValues(list).map(createDf)
    rdd = rdd.groupByKey().mapValues(list).map(createDf)
    target_df = rdd.filter(lambda x: x[0] == target).collect()
    rdd = rdd.filter(lambda x: x[0] != target)
    sim = rdd.map(lambda x: (x[0], x[1].corrwith(target_df[0][1])))
    mean_diff = rdd.map(lambda x: (x[0], x[1]-x[1].mean()))
    unnormalized_ratings_diff = mean_diff.join(sim).map(lambda x: (x[0], x[1][0]*x[1][1]))
    unnormalized_ratings_diff = unnormalized_ratings_diff.values().reduce(lambda x,y: x.add(y, fill_value=0))
    normalization_factor = sim.values().reduce(lambda x,y: x.abs().add(y.abs(), fill_value=0))
    normalized_ratings_diff = unnormalized_ratings_diff / normalization_factor
    result = np.array(target_df[0][1]).mean() + normalized_ratings_diff
    return result.sort_values(by=0, ascending=False)