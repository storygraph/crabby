import math
import os
import re
from typing import List, Tuple, Pattern

import crabby.rel.data as data


class SemvalDatasetLoader:
    _RAW_SENT_LINE_PATTERN = r"^[0-9]+\s+\"(.*)\"$"
    
    _raw_sent_line_pattern: Pattern
    
    def __init__(self) -> None:
        self._raw_sent_line_pattern = re.compile(self._RAW_SENT_LINE_PATTERN)
    
    def load_dataset(self) -> data.SentencePairer:
        data_path = os.getenv("DATA_DIR", "")
        trainset_path = os.path.join(data_path, "relex", "train.txt")
        
        with open(trainset_path, 'r') as stream:
            lines = stream.readlines()

        groups = self._split_into_groups(lines)
        sentences = [None] * len(groups)
        labels = [None] * len(groups)
        
        for i, group in enumerate(groups):
            sent, label = self._split_group(group)
            
            sentences[i] = sent
            labels[i] = label
        
        # Just taking the unique ones
        relations = list(set(labels))
        
        split_idx = math.ceil(len(sentences) * 0.9)
        
        training_pairer = data.SentencePairer(sentences[:split_idx], labels[:split_idx], relations)
        test_pairer = data.SentencePairer(sentences[split_idx:], labels[split_idx:], relations)
        
        return training_pairer, test_pairer

    def _split_into_groups(self, lines: List[str]) -> List[List[str]]:
        groups = [None] * (len(lines) // 4)
        curr = 0
        i = 0

        for line in lines:
            if i == 4:
                i = 0
                curr += 1
            
            if i == 0:
                groups[curr] = []
            
            if i < 2:
                groups[curr].append(line.rstrip())
            
            i += 1

        return groups

    def _split_group(self, group: List[str]) -> Tuple[str, str]:
        raw_sent = group[0]
        # guaranteed to be one.
        sent = re.findall(self._raw_sent_line_pattern, raw_sent)[0]
        
        return sent, group[1]
