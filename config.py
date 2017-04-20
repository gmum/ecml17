import logging
import os
from dotenv import load_dotenv, find_dotenv

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO)

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())

repo_path = os.path.dirname(os.path.realpath(__file__))

config = {
    'data_dir': os.path.join(repo_path, 'data'),
    'results_dir': os.environ.get('RESULTS_DIR')
}

logging.info("Configured env variables: {}".format(str(config)))