"""
BertLogParser: A log parser that uses BERT embeddings to generate log keys.
This class mimics the Drain algorithm’s interface by implementing the two
important methods: outputResult and parse. Instead of using tree-based clustering,
BERT is used to create and update log clusters based on embedding similarity.
"""

import regex as re
import os
import numpy as np
import pandas as pd
import hashlib
import uuid
from datetime import datetime
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, ModernBertModel
import torch

class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/', st=0.2, drift_threshold=0.3, rex=[], keep_para=True):
        """
        Parameters
        ----------
        log_format : str
            The log format string (used to generate a regex for splitting).
        indir : str
            Input directory where the log file is stored.
        outdir : str
            Output directory to store structured logs.
        st : float
            Similarity threshold for assigning a log entry to an existing cluster.
        drift_threshold : float
            If the distance to a cluster center exceeds this threshold, a new cluster is created.
        rex : list
            A list of regular expressions for preprocessing the log messages.
        keep_para : bool
            Whether to extract and keep parameter lists.
        """
        self.path = indir
        self.savePath = outdir
        self.log_format = log_format
        self.rex = rex
        self.keep_para = keep_para
        self.logName = None
        self.df_log = None
        
        # Initialize BERT model and tokenizer from Hugging Face
        #self.tokenizer = BertTokenizer.from_pretrained('modernbert-base')
        #self.model = BertModel.from_pretrained('modernbert-base')
        
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.model = ModernBertModel.from_pretrained("answerdotai/ModernBERT-base")
        
        self.model.eval()  # Set to evaluation mode
        
        # In-memory storage for clusters.
        # Each key maps to a dict with keys: 'embeddings', 'center', 'logIDs', 'template'
        self.clusters = {}
        
        # Similarity thresholds for cluster assignment and drift
        self.similarity_threshold = st
        self.drift_threshold = drift_threshold

    def get_embedding(self, text):
        #print(text)
        """Generate a BERT embedding for the given text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the [CLS] token embedding as the representation
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
        # Normalize the embedding (helpful for cosine similarity)
        norm_embedding = cls_embedding / np.linalg.norm(cls_embedding)
        return norm_embedding

    def cosine_distance(self, a, b):
        """Compute cosine distance between two vectors."""
        return 1 - np.dot(a, b)

    def assign_embedding(self, text, logID):
        """
        Process a log message to generate its embedding and assign it to a cluster.
        If an existing cluster is similar enough, update that cluster; otherwise, create a new cluster.
        """
        embedding = self.get_embedding(text)
        assigned_cluster = None
        # Search existing clusters
        for cluster_id, cluster in self.clusters.items():
            center = cluster['center']
            dist = self.cosine_distance(embedding, center)
            #print(f"Distance to cluster {cluster_id}: {dist}")
            if dist < self.similarity_threshold:
                # Check for drift: if the embedding is too far from the cluster center, skip
                if dist > self.drift_threshold:
                    continue
                assigned_cluster = cluster_id
                # Update cluster by appending the new embedding and log line ID
                cluster['embeddings'].append(embedding)
                cluster['logIDs'].append(logID)
                # Update cluster center as the mean of all embeddings in the cluster
                cluster['center'] = np.mean(cluster['embeddings'], axis=0)
                # Optionally, keep the first seen log message as the cluster template
                break
        if assigned_cluster is None:
            # Create a new cluster with a new log key (UUID)
            new_cluster_id = str(uuid.uuid4())
            self.clusters[new_cluster_id] = {
                'embeddings': [embedding],
                'center': embedding,
                'logIDs': [logID],
                'template': text  # Use this log message as the cluster's template
            }
            assigned_cluster = new_cluster_id
        return assigned_cluster

    def preprocess(self, line):
        """Preprocess a log line by applying regex substitutions."""
        
        regex_replacements = {
            # IPv4 address pattern – replaces something like "192.168.1.100" with "ip_address"
            r'(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)': 'ip address',

            # Domain names with optional ports – replaces "example.com:8080" with "domain"
            r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?::\d{1,5})?\b': 'domain',

            # Any standalone number – replaces numbers like "123" with "number"
            r'\b\d+\b': 'number',

            # File sizes – matches terms like "KB", "MB", "GB", etc.
            r'\b[KMGT]?B\b': 'filesize',

            # Unix-style file paths – matches paths like "/var/log/app.log"
            r'\/(?:[\w.-]+\/)*[\w.-]+': 'filepath',

            # Windows-style file paths – matches paths like "C:\Program Files\App\app.exe"
            r'[A-Za-z]:\\(?:[^\\\/:*?"<>|\r\n]+\\)*[^\\\/:*?"<>|\r\n]+': 'filepath',

            # Timestamps – matches common timestamp formats such as "2021-08-01 12:34:56"
            r'\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\b': 'timestamp',

            # URLs – replaces web addresses with "url"
            r'https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)': 'url',

            # UUIDs – matches universally unique identifiers
            r'\b[0-9a-fA-F]{8}\b-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b': 'uuid'
        }
        
        for pattern, replacement in regex_replacements.items():
            line = re.sub(pattern, replacement, line)
        return line

    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def load_data(self):
        """
        Load log file data into a pandas DataFrame.
        This is similar to Drain’s load_data method.
        """
        headers, regex_obj = self.generate_logformat_regex(self.log_format)
        log_messages = []
        linecount = 0
        file_path = os.path.join(self.path, self.logName)
        with open(file_path, 'r') as fin:
            for line in fin.readlines():
                try:
                    match = regex_obj.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception:
                    pass
        self.df_log = pd.DataFrame(log_messages, columns=headers)
        self.df_log.insert(0, 'LineId', [i + 1 for i in range(linecount)])
        print("Total lines: ", len(self.df_log))

    def get_parameter_list(self, row):
        """
        Mimics Drain’s parameter extraction.
        """
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex:
            return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r'\\ +', r'\\s+', template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return parameter_list

    def outputResult(self):
        """
        Generate output files (structured logs and templates) similar to Drain’s outputResult.
        Each cluster’s template is used to generate an EventId (via MD5 hash).
        """
        log_templates = [0] * self.df_log.shape[0]
        log_templateids = [0] * self.df_log.shape[0]
        df_events = []
        for cluster_id, cluster in self.clusters.items():
            # Use the stored template (first seen log) as the representative template
            template_str = cluster.get('template', '')
            occurrence = len(cluster['logIDs'])
            event_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for logID in cluster['logIDs']:
                # Adjust for zero-indexed DataFrame rows (LineId is 1-indexed)
                log_templates[logID - 1] = template_str
                log_templateids[logID - 1] = event_id
            df_events.append([event_id, template_str, occurrence])
        self.df_log['EventId'] = log_templateids
        self.df_log['EventTemplate'] = log_templates
        
        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)
        
        # Save the structured logs CSV
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        output_file = os.path.join(self.savePath, self.logName + '_structured.csv')
        self.df_log.to_csv(output_file, index=False)
        
        # Save the templates CSV
        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['EventId'] = df_event['EventTemplate'].map(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8])
        df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)
        output_file2 = os.path.join(self.savePath, self.logName + '_templates.csv')
        df_event.to_csv(output_file2, index=False, columns=["EventId", "EventTemplate", "Occurrences"])

    def parse(self, logName):
        """
        Parse the log file with the given name.
        For each log line, preprocess the content, compute its BERT embedding, and assign it a log key.
        Then output the results.
        """
        print('Parsing file: ' + os.path.join(self.path, logName))
        start_time = datetime.now()
        self.logName = logName
        self.load_data()
        
        # For each log entry, combine all columns (except LineId) as the log message content.
        # Preprocess the message and assign it to a cluster using the BERT embedding.
        for idx, row in self.df_log.iterrows():
            logID = row['LineId']
            # Concatenate all non-LineId fields to form the log content
            processed_content = self.preprocess(row['Content']).strip()
            # Get or create a cluster for this log entry
            cluster_id = self.assign_embedding(processed_content, logID)
            # Optionally, store the BERT log key in the DataFrame for later analysis
            self.df_log.loc[idx, 'BertLogKey'] = cluster_id
        
        self.outputResult()
        print('Parsing done. [Time taken: {}]'.format(datetime.now() - start_time))



# Example usage:
if __name__ == "__main__":
    # Example log format (customize as needed)
    log_format = "<Date> <Time> <Pid> <Level> <Content>"
    # Optional regex patterns to mask dynamic parts (e.g., IP addresses, numbers, etc.)
    rex = [r'\d+\.\d+\.\d+\.\d+', r'\d+']
    parser = LogParser(log_format=log_format, indir='./logs', outdir='./result', st=0.2, drift_threshold=0.3, rex=rex)
    # Replace 'system.log' with your actual log filename
    parser.parse('system.log')
