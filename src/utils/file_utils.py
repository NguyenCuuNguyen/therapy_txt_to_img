import os
import random
import logging

class FileUtils:
    """Utility class for file operations and sampling."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def sample_transcripts(self, transcript_dir, sample_size, output_path):
        """Sample random transcripts and save the list."""
        transcript_files = [f for f in os.listdir(transcript_dir) if f.endswith('.txt')]
        if len(transcript_files) < sample_size:
            raise ValueError(f"Requested {sample_size} transcripts, but only {len(transcript_files)} available.")
        
        sampled_files = random.sample(transcript_files, sample_size)
        sampled_filenames = [os.path.splitext(f)[0] for f in sampled_files]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(sampled_filenames))
        self.logger.info(f"Saved sampled transcript list to {output_path}")
        return sampled_filenames

    def load_sampled_transcripts(self, sample_list_path):
        """Load list of sampled transcript filenames."""
        try:
            with open(sample_list_path, 'r') as f:
                return [line.strip() for line in f]
        except Exception as e:
            self.logger.error(f"Failed to load sample list {sample_list_path}: {e}")
            raise