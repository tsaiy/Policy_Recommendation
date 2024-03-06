# Conservatives Agricultural Policy Recommendation System (APRS) 

Traditional agricultural policies are often reactive, responding to market shifts and environmental changes with a delayed and sometimes inadequate approach. This paper aims to address the critical need for an innovative APRS that leverages advanced forecasting techniques to provide real-time insights into the upcoming trends in agricultural produce demand and supply. By doing so, the system will assist policymakers, farmers, and other stakeholders in proactively shaping policies that align with the evolving dynamics of the agricultural landscape. The Conservatives Agricultural Policy Recommendation System (APRS) is an innovative decision support system designed to address the complex challenges faced by modern agriculture. By leveraging advanced forecasting techniques, the APRS provides real-time insights into the demand and supply dynamics of agricultural produce, empowering stakeholders to make informed and proactive policy decisions. Through the development and implementation of such a system, the app seeks to contribute to the evolution of agricultural policies, transforming them from reactive measures to proactive strategies that anticipate and adapt to the ever-changing dynamics of the agricultural sector. The successful deployment of an APRS grounded in demand-supply forecasts will not only enhance the efficiency and resilience of agricultural systems but also promote sustainable practices and contribute to the overall growth and stability of the agricultural economy. Traditional agricultural policies are often reactive and insufficiently responsive to market shifts and environmental changes. There is a critical need for a comprehensive APRS that anticipates trends and assists policymakers, farmers, and other stakeholders in shaping proactive strategies aligned with the evolving agricultural landscape. 


### Objectives 


- Develop an innovative APRS grounded in demand-supply forecasts.
- Empower stakeholders with real-time insights for policy formulation and implementation.
- Foster sustainable agricultural practices and promote economic growth and stability. 


### Potential Use Cases: 

1. Government and Policy Making: Government agencies can formulate evidence-based policies with targeted subsidy allocation.
2. Supply Chain Management: Agribusinesses optimize distribution networks and inventory management for efficient supply chains.
3. Financial Institutions: Financial institutions use APRS insights for assessing investment viability and making informed lending decisions.
4. International Development Organizations: Organizations enhance food security programs by using APRS insights for targeted interventions.

### Novel Machine Learning Techniques 

The APRS leverages novel machine learning techniques, including Long Short-Term Memory (LSTM) and several decision trees to perform univariate and mulivariate forecasting, to analyze historical data, predict future trends, and recommend policy interventions. These models are then orchestrated on a langchain environment to create a funnel mechanism that leads to policy generation for the given input. 

![Architecture-ASPR](https://github.com/Naveen-Nanda/Policy_Recommendation/assets/29003849/06fdde8d-ff75-4244-9913-7d96fc060c91)

### Project Structure The project is organized into the following components:

1. **Scripts**: Contains Python scripts for various tasks, including data preprocessing, model training, and application deployment. 
2. **Data**: Stores datasets used for training and evaluation. 
3. **Models**: Stores trained machine learning models and associated files. 
4. **Documentation**: Contains project documentation, including README files and technical specifications. 

### Instructions to Run 

To run the Conservatives APRS, follow these steps: 
1. Install dependencies by running `scripts/install_dependencies.py`.
2. Start the application by executing `scripts/launch_app.py`.
3. Access the application through the provided URL or local host. Ensure that you have the necessary permissions and resources to execute the scripts and deploy the application. 


Dataset Google Drive - [Link](https://drive.google.com/drive/folders/1Aysc8DK8vkETSP-JF0DpmwONkS7wpuRX?usp=sharing) 


App Demo Link - https://policyrecommendation-cloudera.streamlit.app/
