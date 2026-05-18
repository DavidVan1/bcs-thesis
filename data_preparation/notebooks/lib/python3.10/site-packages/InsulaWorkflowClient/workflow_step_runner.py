import types
from os.path import basename, join
from multiprocessing.pool import ThreadPool
from snakenest import Poisoned
from .InsulaApiConfig import InsulaApiConfig
from .job_params import InsulaJobParams
from .workflow_step import InsulaWorkflowStep
from .job_runner import InsulaRunner
from .job_logs import InsulaJobLogs
from .files_job_result import InsulaFilesJobResult
from .downloadJob_results import InsulaDownloadJobResults
from .utils import InsulaUtils
from .step_result import StepResult
from .job_status import InsulaJobStatus
from .WorkflowDataManager import WorkflowData
from .results_manager import ResultManager
from .placeholders import Placeholder
from .delete import InsulaDeleter
from .logger import logger


class InsulaWorkflowStepRunner(object):
    @Poisoned()
    def __init__(self,
                 insula_config: InsulaApiConfig,
                 steps: InsulaWorkflowStep,
                 workflow_data: WorkflowData,
                 result_manager: ResultManager,
                 ):

        super().__init__()
        self.__insula_api_config = insula_config
        self.__steps = steps
        self.__workflow_data = workflow_data
        self.__result_manager = result_manager

        self.__max_parallel_jobs = self.__workflow_data.config['max_parallel_jobs']

    def __translate_values(self, param: dict) -> list[StepResult]:
        job_results = []
        for value in [param['values']] if isinstance(param['values'], str) else param['values']:
            job_results.extend(Placeholder(value, self.__workflow_data).get_results())

        return job_results

    # TODO: da implementare continue_on_error nello step
    # TODO: cambiare il run con un oggetto
    def __run_platform_processor_step(self, step):
        # attempt = 0
        run = {
            'name': step['name'],
            'service_id': self.__insula_api_config.get_platform_service_url_api_path(step['service_id'])
        }

        # TODO: This retry management should be removed because it is currently performed inside InsulaRunner
        # while attempt < self.__insula_api_config.max_processor_attempts:
        #     logger.info(f'Attempt: {attempt}\n\tstep: {step}')

        insula_job_params = InsulaJobParams(run['service_id'])
        for param in step['params']:
            params_arr = []
            for _ in self.__translate_values(param):
                params_arr.append(_.get('default'))

            insula_job_params.set_inputs(str(param['name']), params_arr)

        insula_status = InsulaRunner(self.__insula_api_config).run(insula_job_params)

        run['status'] = insula_status.get_status()

        # TODO: sta cosa e' orrenda
        if run['status']['status'] != 'COMPLETED':
            logger.info(run['status'])
            insula_logs = InsulaJobLogs(self.__insula_api_config)
            logs = insula_logs.get_logs(insula_status.get_job_id())
            if logs is None:
                run['logs'] = {}
            else:
                run['logs'] = logs
        else:
            run['results'] = InsulaFilesJobResult(self.__insula_api_config).get_result_from_job_status(insula_status)

        return run

    def __run_load_result_step(self, step):
        # TODO: the step_resoult should be a class
        run = {
            'name': step['name'],
            'service_id': 0,
            'status': {
                "config_id": 0,
                "job_id": 0,
                "status": "REQUIREMENTS_RETRIEVED"
            },
            'results': []
        }

        run['status']['status'] = 'COMPLETED'
        for param in step['params']:
            for value in self.__translate_values(param):
                run['results'].extend(
                    InsulaFilesJobResult(self.__insula_api_config).get_result_from_job(value.get('default')))

        return run

    def __run_delete(self, step):
        run = {
            'name': step['name'],
            'service_id': 'None',
            'status': {},
            'results': [],
            'logs': {'logs': []}
        }

        run['status']['status'] = 'COMPLETED'

        if 'kind' in step:
            deleter = InsulaDeleter(self.__insula_api_config)
            if step['kind'] == 'platform_file':
                for param in step['params']:
                    try:
                        deleter.delete_files(self.__translate_values(param))
                    except Exception as e:
                        if 'continue_on_error' in step and step['continue_on_error']:
                            pass
                        else:
                            raise Exception(f'cant delete files step {step["name"]} {e}')

        return run

    def __run_download_job_result(self, step):
        # TODO: the step_resoult should be a class
        run = {
            'name': step['name'],
            'service_id': 'None',
            'status': {},
            'results': [],
            'logs': {'logs': []}
        }

        run['status']['status'] = 'COMPLETED'

        downloader = InsulaDownloadJobResults(self.__insula_api_config)
        for param in step['params']:

            if 'save_in' in param:
                save_in = param['save_in']

                if 'create_folder_if_not_exits' in step and step['create_folder_if_not_exits']:
                    InsulaUtils.create_folder_if_not_exists(save_in)
            else:
                save_in = '.'

            all_in = self.__translate_values(param)
            for _ in all_in:
                run['logs']['logs'].append(f'downloading: {_} save in {save_in}')
                status = downloader.download_file(_.get('download'), save_in)
                if not status['success']:
                    run['status']['status'] = 'FAILED'
                    if 'continue_on_error' in step and step['continue_on_error']:
                        run['status']['status'] = 'COMPLETED'

                run['logs']['logs'].append(
                    f'status: {status["success"]} message:{status["message"]}  for {_} save in {save_in}')
        return run

    def __run_s3_step(self, step):
        # TODO: the step_resoult should be a class
        run = {
            'name': step['name'],
            'service_id': 'None',
            'status': {},
            'results': [],
            'logs': {'logs': []}
        }

        continue_on_error = False
        if 'continue_on_error' in step and step['continue_on_error']:
            continue_on_error = True

        status = InsulaJobStatus()
        status.set_job_status('COMPLETED')

        if 'connection' not in step:
            status.set_job_error('FAILED', 'Connection not defined')
            run['status'] = status.get_status()
            return run

        if 'action' not in step:
            status.set_job_error('FAILED', "Action not defined")
            run['status'] = status.get_status()
            return run

        action = step['action']
        if action != 'pull' and action != 'push':
            status.set_job_error('FAILED', f'action unknown: {action}')
            run['status'] = status.get_status()
            return run

        s3_connections = self.__workflow_data.requirements['connections'][step['connection']]
        if not s3_connections:
            status.set_job_error('FAILED', f'connection unknown: {step["connection"]}')
            run['status'] = status.get_status()
            return run

        if step['action'] == 'pull':
            for param in step['params']:
                if 'save_in' in param:

                    # save_in_c = (self.__workflow_data.parse_match(param['save_in']).get_param_changed())

                    # TODO: da sistemare!
                    save_in_c = (Placeholder(param['save_in'], self.__workflow_data).get_results())
                    save_in = save_in_c[0].get('default') if len(save_in_c) > 0 else param['save_in']

                    if 'create_folder_if_not_exits' in step and step['create_folder_if_not_exits']:
                        InsulaUtils.create_folder_if_not_exists(save_in)
                else:
                    save_in = '.'

                for file in self.__translate_values(param):
                    file_name = basename(file.get('default'))
                    try:
                        s3_connections['connection'].download(file.get('default'), join(save_in, file_name))
                    except Exception as e:
                        if not continue_on_error:
                            status.set_job_error('FAILED', e)
                            run['status'] = status.get_status()
                            logger.info(e)
                            return run
        elif step['action'] == 'push':
            for param in step['params']:
                if 'save_in' in param:
                    for file in self.__translate_values(param):
                        try:
                            # TODO: da sistemare!
                            # save_in_c = (self.__workflow_data.parse_match(param['save_in'])
                            #              .get_param_changed())
                            save_in_c = (Placeholder(param['save_in'], self.__workflow_data).get_results())

                            save_in = save_in_c[0].get('default') if len(save_in_c) > 0 else param['save_in']
                            s3_connections['connection'].upload(file.get('default'),
                                                                '/'.join([save_in, basename(file.get('default'))]))
                        except Exception as e:
                            if not continue_on_error:
                                status.set_job_error('FAILED', e)
                                run['status'] = status.get_status()
                                return run

        run['status'] = status.get_status()
        return run

    # https://stackoverflow.com/questions/701802/how-do-i-execute-a-string-containing-python-code-in-python
    def __run_python_script(self, step):
        # TODO: the step_resoult should be a class
        run = {
            'name': step['name'],
            'service_id': 'None',
            'status': {},
            'results': [],
            'logs': {'logs': []}
        }

        continue_on_error = False
        if 'continue_on_error' in step and step['continue_on_error']:
            continue_on_error = True

        status = InsulaJobStatus()
        status.set_job_status('COMPLETED')

        my_namespace = types.SimpleNamespace()
        for param in step['params']:

            return_values = self.__translate_values(param)
            if len(return_values) > 1:
                my_namespace.__setattr__(param['name'], [])
                for __ in return_values:
                    if isinstance(__, list):
                        for _ in __:
                            my_namespace.__getattribute__(param['name']).append(_.get('default'))
                    else:
                        my_namespace.__getattribute__(param['name']).append(__.get('default'))
            else:
                if len(return_values) == 1:

                    if isinstance(return_values[0], list):
                        my_namespace.__setattr__(param['name'], [])
                        for _ in return_values[0]:
                            my_namespace.__getattribute__(param['name']).append(_.get('default'))
                    else:
                        my_namespace.__setattr__(param['name'], return_values[0].get('default'))
                else:
                    my_namespace.__setattr__(param['name'], None)

        try:
            exec(step['source'], my_namespace.__dict__, )
        except Exception as e:
            if not continue_on_error:
                status.set_job_error('FAILED', e)
                run['status'] = status.get_status()
                return run

        outputs_attr = [x for x in dir(my_namespace) if not (x.startswith('__') and x.endswith('__'))]
        if 'outputs' in step:
            for output in [step['outputs']] if isinstance(step['outputs'], str) else step['outputs']:
                if output in outputs_attr:
                    att = my_namespace.__getattribute__(output)
                    if isinstance(att, str):
                        run['results'].append(StepResult(id=output, output_id=output, default=att, type='script'))
                    elif isinstance(att, list):
                        for __ in att:
                            run['results'].append(StepResult(id=output, output_id=output, default=__, type='script'))
                    else:
                        if not continue_on_error:
                            status.set_job_error('FAILED',
                                                 f'Invalid output type {output}, str and list are not supported')
                            run['status'] = status.get_status()
                            return run

        run['status'] = status.get_status()
        return run

    def __run(self, step):

        if 'type' in step:
            if step['type'] == 'processor':
                run = self.__run_platform_processor_step(step)
            elif step['type'] == 'results':
                run = self.__run_load_result_step(step)
            elif step['type'] == 's3':
                run = self.__run_s3_step(step)
            elif step['type'] == 'script':
                run = self.__run_python_script(step)
            elif step['type'] == 'delete':
                run = self.__run_delete(step)
            elif step['type'] == 'downloader':
                run = self.__run_download_job_result(step)
            else:
                raise Exception(f'Unknown step: {step["type"]}')
        else:
            raise Exception(f"type not defined in step {step['name']}")

        return {
            'run': run,
            'step': step,
            'error': run['status']['status'] != 'COMPLETED'
        }

    def run(self):
        results = []

        step_count = self.__steps.count()
        if step_count == 1:
            results = [self.__run(self.__steps.get_step(0))]
        else:
            with ThreadPool(processes=self.__max_parallel_jobs) as pool:
                results = pool.map(self.__run, self.__steps.get_steps())

        are_there_errors = False
        for result in results:
            if result['error']:
                are_there_errors = True

        return {
            'results': results,
            'error': are_there_errors
        }
