# Automating Customer Support Issue Classification with Fine-Tuned Large Language Models (LLMs)

## Overview
This README provides a comprehensive guide on leveraging open source Large Language Models (LLMs) to automate the classification of customer support issues. By fine-tuning a pre-trained LLM on a custom dataset, organizations can accurately predict Task Types for customer support requests, thereby streamlining customer service operations and reducing support center costs.

## Importance of Classifying Customer Support Calls
Understanding caller-to-agent interactions in customer support is crucial for improving service quality and reducing costs. Task Types represent distinct reasons why customers contact support, such as technical issues, product inquiries, or billing problems. Accurate classification of Task Types allows organizations to identify recurring patterns, improve self-service options, and enhance customer satisfaction.

## Automating Customer Support Issue Classification with Fine-Tuned LLMs
In this tutorial, we'll demonstrate how to automate Task Type classification using open source LLMs. We'll start with a general LLM and fine-tune it on a custom dataset to improve performance for customer support tasks.

### Overview of Experiments
We'll explore two approaches for fine-tuning the LLM:
1. Using Ludwig: An open-source framework for training custom LLMs.
2. Using Predibase: An enterprise platform for fine-tuning and serving LLMs on managed infrastructure.

## Preparing the Dataset
We'll use the Gridspace-Stanford Harper Valley (GSHV) dataset, which contains annotated transcripts of customer support conversations. The dataset includes both transcript and metadata directories, each containing files with relevant information such as speaker roles, transcripts, and Task Types.

## Fine-Tuning Zephyr-7B-Beta with Ludwig
We'll first establish a baseline using the Zephyr-7B-Beta LLM and then fine-tune it on the GSHV dataset using Ludwig. This involves configuring Ludwig with input and output features, defining prompts, and training the model. We'll evaluate the fine-tuned model's performance using accuracy metrics.

## Fine-Tuning Zephyr-7B-Beta with Predibase
Next, we'll fine-tune the Zephyr-7B-Beta LLM on the GSHV dataset using Predibase. After uploading the dataset to Predibase cloud, we'll launch a fine-tuning job and monitor its progress. Once complete, we'll retrieve the fine-tuned model and evaluate its performance.

## Conclusion
Automating Task Type classification in customer support calls using fine-tuned LLMs offers significant benefits in terms of efficiency and cost-effectiveness. By following this tutorial, people can integrate these techniques into their customer support systems to enhance service quality and reduce operational costs.
