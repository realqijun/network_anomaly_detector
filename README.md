## Network Anomaly Detector

### Description

Network Anomaly Detector is an engine I trained to label network packets on whether they are Anomalous (1) or Benign (0). It is hosted on a simple Flask app, simply upload a packet capture file(pcap, pcapng), the model will process it, and then display the numerical results.

---

### Project Details

This project is structured as so:
- Main directory has modules used to
  - Take raw pcap file and turn it into a pandas dataframe, then cleaning it up (removing useless columns)
  - Store model
  - Train the model based on the parsed dataframe and determining a good threshold
  - Run tests on the training data
- Datasets directory contains the datasets
  - The dataset is too large for GitHub, but if you wish to obtain the actual datasets, they are opensourced, refer to [references](#references)
- Working directory stores
  - Model weights
  - Threshold
  - Processed data
  - Roc curve graph
- Deploy directory
  - Contains the bare minimal files required for deployment, such as the model weights, model code, and html templates
  - Has Dockerfile that runs the build, which uses requirements.txt to install the neccessary libraries

This was trained using an autoencoder, which is a type of neural network that takes in unlabelled data, creates a model of it, and is checked against mixed data. This allows it to label outliers effectively, by setting a reasonable threshold (in this case, an AUC-ROC score of above 0.80). Note that the unlabelled data should be mostly benign so that the model knows what is considered "normal".

This project was started to try to game the network packet analysis CTFs, where I can quickly find out which packets may contain useful data on the flag. Later I found out that the rate of false positives were too high and further filter and analysis of the anomalous packets.

---

### Limitations

1. The program currently only supports "manual" detection, meaning that you would have to capture packets on your own, and upload it to the website to check for anomalies. A planned enhancement for this is to make it a program that runs locally and has access to the Network Interface Card (NIC), which allows the program to capture packets and label it in real time.

2. As this is only an anomaly detector, false positives are expected. Although dataset used to train the model is rather large (even though the training was extensive), the AUC-ROC score of 0.80 although considered decent, resulted in thousands of false positives, leading to alert fatigue. Nonetheless, more training is still needed as network packets can be very complex and have many features.

---

The model was trained using the Intrusion detection evaluation dataset (CIC-IDS2017).

### References

1. Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”, 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018.

**If you have any enquiries, or suggestions, I would love it hear it! Kindly reach out to me**
