# Accomodation Booking prediction

### contributors
- Mayank Raghav (G23AI1022)
- Abhinava Bharat Gavireddi (G23AI1012)
- Nachiket Mahishi (G23AI1024)

## Architecture


![infrastructure drawio](https://github.com/user-attachments/assets/86f2f23c-26d4-432e-b897-985d88ab000f)



Details of the project:

- We experimented with Kinesis to ingest streaming data and save to S3.
- Considering the cost of kinesis, we experimented with Kafka by deploying on EC2 instance, but was unable to connect to local producer (WIP).
- So considering time and cost, we uploaded the raw data in our S3 bucket.
- We created IAM role and gave it S3 access permissions to streamlit API which is deployed on HuggingFace.
- Retention policies were applied to S3 bucket after a time to transfer the data to S3 glacier to save storage costs.
- Life cycle policies were also applied for an year to both Glacier and Standard Bucket.
- We created an S3 event trigger which invokes step function. The step function definition consists of calling a glue job to transform data and finally running the crawler.
- The crawler created the meta data in the AWS glue catalog.
- Used Athena to Query the data in AWS glue catalog.
- Adjust data for seasonality and create demand forecasting models.
- Use machine learning to predict booking cancellations.

## Deployment

- For deployment we created a cloud formation template that creates all the resources that are mentioned above, we connected that to AWS code deploy and we tried to deploy the resources, to reduce costs, we resorted to creating the resources from UI and from CLI.



-: The project is deployed in [here](https://huggingface.co/spaces/mayankraghav/Capstone_project) :- 
