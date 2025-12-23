# Google Maps Recommender System

## Overview

We developed a recommender system using the Google Maps review dataset to recommend places (POIs) that users are likely to enjoy. We employed **LightGCN (Light Graph Convolutional Network)** [[1]](#ref1), a state-of-the-art graph neural network (GNN) model.
This project allows you to create your own subset of review data by specifying a list of categories to extract, enabling the construction of a customized recommender system tailored to specific domains.

## Dataset

### Raw Dataset
We used [Google Local Data (2021)](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/) [[2](#ref2), [3](#ref3)], which contains approximately 660 million reviews, 110 million users, and 4 million businesses in the United States (up to September 2021).

The dataset consists of two types of tables: `review` and `meta`. Below are the descriptions of the columns specifically used in our model:

* **`review`** (User-Business Interactions):
    * `user_id`: Unique user ID.
    * `gmap_id`: Unique business ID.
    * `name`: User name (used only for inference).
    * `time`: Unix timestamp (in milliseconds) of the review.
    * `rating`: Discrete rating from 1 to 5.

* **`meta`** (Business Information):
    * `gmap_id`: Unique business ID.
    * `name`: Business name (used only for inference).
    * `address`: Business address.
    * `latitude` / `longitude`: Geolocation of the business.
    * `category`: Business categories (used for filtering).
    * `avg_rating`: Average rating (used for analysis).
    * `num_of_reviews`: Total number of reviews.
    * `price`: Price range scale (e.g., "\$", "\$\$").

### Filtering
We first filtered businesses and reviews based on specified categories using the `category` column in the metadata.
To address the data sparsity problem, we conducted **$k$-core filtering** with $k=5$. This ensures that all users and businesses in the dataset have at least $k$ associated reviews.

In our experiments, we constructed three distinct datasets to evaluate our model:

| Dataset | Categories | States | Users | Businesses | Reviews |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Japanese** | Japanese, Ramen, Sushi | All | 136,490 | 21,725 | 972,295 |
| **Cafe** | Cafe, Coffee | CA (California) | 139,856 | 13,472 | 1,215,589 |
| **Asian** | Chinese, Japanese, Thai, Vietnamese, Korean, Indian, Filipino | All | 549,269 | 72,226 | 4,638,057 |

### Train/Validation/Test Splits (Temporal Split)

We split the dataset into three sets using a **leave-one-out strategy** based on timestamps.
For each user, we sorted their reviews chronologically and assigned them as follows:

* **Test:** The most recent review.
* **Validation:** The second most recent review.
* **Train:** All remaining historical reviews.

Since we applied 5-core filtering, every user has at least 5 reviews. This guarantees that the training set contains at least 3 reviews per user, while the validation and test sets contain exactly one review per user.

## Model

### Building Graph

### LightGCN

### Techniques


## Inference

## Analysis


## Customization

## References
## References

1.  <a id="ref1"></a> **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation**
    Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, & Min-Yen Kan.
    *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '20)*.
    [[Paper](https://arxiv.org/abs/2002.02126)]

2.  <a id="ref2"></a> **UCTopic: Unsupervised Contrastive Learning for Phrase Representations and Topic Mining**
    Jiacheng Li, Jingbo Shang, Julian McAuley.
    *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL '22)*.
    [[Paper](https://aclanthology.org/2022.acl-long.426/)]

3.  <a id="ref3"></a> **Personalized Showcases: Generating Multi-Modal Explanations for Recommendations**
    An Yan, Zhankui He, Jiacheng Li, Tianyang Zhang, Julian McAuley.
    *Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '23)*.
    [[Paper](https://arxiv.org/abs/2305.16643)]