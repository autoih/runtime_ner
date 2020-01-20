import io
import os
import re
import shutil
import time
import zipfile
from urllib.parse import urlparse

import yaml

import watson_nlp
from watson_nlp.toolkit import web, performance


class PerformanceBase:
    # PyTest doesn't pick up test files with __init__
    perf_artifacts = 'perf_test_artifacts'
    perf_data = perf_artifacts + '/data'
    results_file = perf_artifacts + '/results'
    perf_models = perf_artifacts + '/models/'
    load_times = {}

    def get_model(self, model_name, language):
        '''Download and return model for the given model_name and language, tracking its load time
        '''
        os.makedirs(PerformanceBase.perf_models, exist_ok=True)
        downloaded_model = watson_nlp.download(model_name, parent_dir=PerformanceBase.perf_models)
        start = time.time()
        loaded_model = watson_nlp.load(downloaded_model)
        load_time = time.time() - start

        # Maintain block and language level load times
        self.load_times[loaded_model.BLOCK_NAME] = self.load_times.get(loaded_model.BLOCK_NAME, {language: 0})
        self.load_times[loaded_model.BLOCK_NAME][language] = load_time
        return loaded_model

    @staticmethod
    def pull_test_data(language):
        '''Pull language specific test data set if not already saved in default path.
        Returns dict with description and test_data path
        '''
        paren_data = "{0}/{1}".format(PerformanceBase.perf_data, language)
        os.makedirs(paren_data, exist_ok=True)  # For Jenkins builds
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test_data.yml'), 'r') as test_data:
            test_data_specs = yaml.safe_load(test_data)['specifications']
            data_set = []
            # Test data specifications data for multiple languages
            for language_data in test_data_specs:
                if language_data['language'] == language:
                    benchmarks = language_data['benchmarks']
                    # Individual language can have multiple data sets
                    for benchmark in benchmarks:
                        # Data artifactory url
                        test_data_url = benchmark['link']
                        is_zip = test_data_url.endswith('.zip')
                        test_data_description = benchmark['description']

                        # We support compressed archives (.zip) which are expanded
                        # and text files which are downloaded as their saved basename
                        data_set_name = os.path.basename(urlparse(test_data_url).path)
                        test_data_dir = "{0}/{1}".format(paren_data, data_set_name)

                        test_data_dir = os.path.splitext(test_data_dir)[0] if is_zip else test_data_dir
                        if os.path.exists(test_data_dir):
                            print("{} already saved".format(test_data_dir))
                        else:
                            response, _ = web.WebClient.request(test_data_url,
                                                                os.environ.get('ARTIFACTORY_USERNAME'),
                                                                os.environ.get('ARTIFACTORY_API_KEY'))

                            if is_zip:
                                # For zip data archive, expand all files
                                with zipfile.ZipFile(io.BytesIO(response.read())) as model_zip:
                                    model_zip.extractall(test_data_dir)
                            else:
                                # For text file, simply copy it to test folder
                                with open(test_data_dir, 'wb') as out_file:
                                    shutil.copyfileobj(response, out_file)
                        data_set.append({'description': test_data_description, 'test_data': test_data_dir})
            return data_set

    def block_performance(self, benchmark, language, block_arg, notes='', num_seconds=30):
        '''Run performance runner on a given data set for a block tuple and add the generated report to a
        text file in perf_test_artifacts/ which follows the format result_model_language_dataset
        Args:
            benchmark: dict('test_data': str, 'description': str)
                test_data: Path to test benchmark data
                description: Description of test data
            language: str
                Language of the test data set, model
            block_arg: tuple(watson_nlp.blocks.BlockBase, function)
                Tuple with block to test and a pre_process function to transform the raw input
            notes: str
                Optional Notes to add to results title
            num_seconds: int (default - 30)
                Max duration for the performance run on a given model (pre processing times aren't counted)
        '''
        block, pre_process = block_arg
        perf_report = self.function_performance(benchmark, block.run, pre_process, notes, num_seconds)
        self.generate_report(block.BLOCK_NAME,
                             perf_report,
                             language,
                             self.load_times[block.BLOCK_NAME][language])

    @staticmethod
    def function_performance(benchmark, eval_function, pre_process, notes='', num_seconds=30):
        '''Runs timed performance run for any given function
        Args:
            benchmark: dict('test_data': str, 'description': str)
                test_data: Path to test benchmark data
                description: Description of test data
            eval_function: function
                function to perform the test on
            pre_process: function (Optional)
                Function to pre-process raw text. By default raw text is passed to the eval_function
            notes: str
                Optional Notes to add to results title
            num_seconds: int (default - 30)
                Max duration for the performance run on a given model (pre processing times aren't counted)
        Returns:
            watson_nlp.toolkit.performance.PerformanceReport
                Performance report with all calculated metrics (chars, throughput, duration)
        '''
        return performance.PerformanceRunner.timed_function(
            function_arg=(eval_function, pre_process),
            test_directory=benchmark['test_data'],
            num_seconds=num_seconds,
            description=benchmark['description'],
            notes=notes)

    @staticmethod
    def generate_report(function_name, perf_report, language, load_time=0):
        '''Generate performance reports at the end of performance test suite
        '''

        def snake_case(name):
            '''It's not strict snake_case but does a limited conversion'''
            return re.sub(r'([/@,#+])', '', name).replace(' ', '_').lower()

        model_report = "*{}* - *{}* ({}) ".format(function_name, language,
                                                  perf_report.description or 'FAILED')
        model_report += "{}\n".format(perf_report.notes)
        model_report += "```\n{:<13s}: {:<10.3f} {:<13s}: {:<10d}" \
                        "\n{:<13s}: {:<10d} {:<13s}: {:<10.3f}" \
                        "\n{:<13s}: {:<10.3f} {:<13s}: {:<10.3f}\n```" \
            .format('Duration', perf_report.duration,
                    'Characters', perf_report.chars,
                    'Documents', perf_report.docs,
                    'Pre-Process', perf_report.pre_processing_duration,
                    'Throughput', perf_report.throughput,
                    'Load Time', load_time)
        report_name = "{0}_{1}_{2}_{3}".format(PerformanceBase.results_file,
                                               snake_case(function_name),
                                               language,
                                               snake_case(perf_report.description))
        report_name += '_{}'.format(snake_case(perf_report.notes)) if perf_report.notes else ''
        with open(report_name, 'w+') as report_file:
            report_file.write(model_report)