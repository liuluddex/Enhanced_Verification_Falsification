import os
import re
import json
import subprocess
import sys

import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import scienceplots
from tqdm import tqdm

import sys
sys.path.append('.')
sys.path.append('../')

from tools.logs import Logger
from tools.utils import load_plt_data, ReachableSet


class FlowstarExperimentRunner:
    def __init__(self, exp_name: str, target_model_name: str = None):
        super(FlowstarExperimentRunner).__init__()
        self.exp_name = exp_name
        self.target_model_name = target_model_name

        exp_path = os.path.join(os.path.dirname(__file__), '..', 'models', exp_name)
        output_path = os.path.join(exp_path, 'outputs')
        os.system(f'rm -rf {output_path}/*')

        init_config_path = os.path.join(exp_path, 'init.json')
        if os.path.exists(init_config_path):
            self.init_configs = json.load(open(init_config_path))
            if 'num_splits' not in self.init_configs:
                self.num_splits = 1
            else:
                self.num_splits = self.init_configs['num_splits']
                del self.init_configs['num_splits']

        else:
            self.init_configs = {}
            self.num_splits = 1

    def run(self):
        available_models = self.get_available_models()
        Logger.info(f'available models: {available_models}')

        state_ranges = self.generate_state_ranges()

        num_times = 0
        for model_name in available_models:
            for i, state_range in enumerate(state_ranges):
                self.modify_model_inits(model_name, state_range)
                file_paths = self.run_flowstar_model(model_name)
                self.modify_output_name(file_paths, model_name, i + 1)
                num_times += 1

        Logger.info(f'Total times of running {num_times}')

    @staticmethod
    def modify_output_name(file_paths, model_name, idx: int):
        model_name = model_name.replace(".model", "")

        for file_path in file_paths:
            file_name = file_path.split('/')[-1]
            new_file_name = file_name.replace(model_name, f'{idx}-{model_name}')
            new_file_path = file_path.replace(file_name, new_file_name)
            os.system(f'mv {file_path} {new_file_path}')

    def generate_state_ranges(self):
        Logger.info(f'Start generating initial state sets')
        state_range_sets = {}
        for key, value in self.init_configs.items():
            split_values = np.linspace(value[0], value[1], self.num_splits + 1)
            state_range_sets[key] = []
            for i in range(self.num_splits):
                state_range_sets[key].append([split_values[i], split_values[i + 1]])

        state_ranges = []
        for i in range(self.num_splits):
            state_range = {}
            for key in state_range_sets.keys():
                state_range[key] = state_range_sets[key][i]
            state_ranges.append(state_range)

        Logger.info(f'Finish generating initial state sets')

        return state_ranges

    def modify_model_inits(self, model_name, state_range):
        Logger.info(f'Start modifying initial state ranges of model {model_name}')
        exp_path = os.path.join(os.path.dirname(__file__), '..', 'models', self.exp_name)
        model_path = os.path.join(exp_path, model_name)

        state_variable_indexes = []
        lines = open(model_path, 'r', encoding='utf-8').readlines()
        for i in range(len(lines)):
            if re.findall('\[.+\]', lines[i].strip()) and 'in' in lines[i]:
                state_variable_indexes.append(i)

        for index in state_variable_indexes:
            state_var = lines[index].strip().split(' ')[0]
            old_state_set = re.findall('(\[.+\])', lines[index].strip())[0]
            if state_var not in state_range:
                continue

            state_set = f'[{state_range[state_var][0]}, {state_range[state_var][1]}]'
            lines[index] = lines[index].replace(old_state_set, state_set)

        open(model_path, 'w', encoding='utf-8').writelines(lines)
        Logger.info(f'Finish modifying initial state ranges of model {model_name}')

    def run_flowstar_model(self, model_name: str):
        case_name = model_name.replace('.model', '')
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        model_file_path = os.path.join(model_path, self.exp_name, model_name)

        config_path = os.path.join(model_path, self.exp_name, model_name.replace('.model', '.json'))

        if not os.path.exists(config_path):
            return

        config = json.load(open(config_path, 'r', encoding='utf-8'))
        pairs = [item.split('-') for item in config['pairs']]

        tool_path = os.path.dirname(__file__)
        command = f'{tool_path}/flowstar < {model_file_path}'

        Logger.info(f'Start running {model_name} ...')
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        Logger.info(f'Finish running {model_name}')

        Logger.info(f'Start generating flow files ...')

        output_path = os.path.join(os.path.dirname(__file__), 'outputs')
        input_flow_path = os.path.join(output_path, f'{case_name}.flow')
        lines = open(input_flow_path, 'r', encoding='utf-8').readlines()
        num_lines = len(lines)
        index = -1
        for i in range(num_lines):
            if re.match('gnuplot interval', lines[i].strip()):
                index = i
                break

        for i, (x_var, y_var) in enumerate(pairs):
            lines[index] = re.sub('gnuplot interval (.+)', f'gnuplot interval {x_var}, {y_var}', lines[index])
            open(input_flow_path, 'w', encoding='utf-8').write(''.join(lines))

            Logger.info(f'[{i + 1}/{len(pairs)}] Start generating flow file for {x_var}-{y_var} ...')
            command = f'./flowstar < {input_flow_path}'
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
            Logger.info(f'[{i + 1}/{len(pairs)}] Finish generating flow file for {x_var}-{y_var}')

            if not os.path.exists(os.path.join(model_path, self.exp_name, 'outputs')):
                os.mkdir(os.path.join(model_path, self.exp_name, 'outputs'))

            input_plt_path = os.path.join(output_path, f'{case_name}.plt')
            output_plt_path = os.path.join(model_path, self.exp_name, 'outputs', f'{case_name}-{x_var}-{y_var}.plt')
            os.system(f'mv {input_plt_path} {output_plt_path}')
            os.system(f'mv {input_plt_path.replace(".plt", ".flow")} {output_plt_path.replace(".plt", ".flow")}')

        file_paths = []

        for x_var, y_var in pairs:
            file_paths.append(os.path.join(model_path, self.exp_name, 'outputs', f'{model_name.replace(".model", "")}-{x_var}-{y_var}.flow'))
            file_paths.append(os.path.join(model_path, self.exp_name, 'outputs', f'{model_name.replace(".model", "")}-{x_var}-{y_var}.plt'))

        return file_paths

    def get_available_models(self):
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        available_exp_names = [dir_name for dir_name in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, dir_name))]
        if self.exp_name not in available_exp_names:
            Logger.error(f'Experiment {self.exp_name} is not an available experiment!')
            return

        available_models = [model_name for model_name in os.listdir(os.path.join(model_path, self.exp_name)) if
                            model_name.endswith('.model')]
        if self.target_model_name is not None:
            available_models = [model_name for model_name in available_models if self.target_model_name in model_name]

        return available_models


class FlowstarExperimentVisualizer:
    def __init__(self, exp_name: str, target_model_name=None, sim_time=4.0):
        super(FlowstarExperimentVisualizer).__init__()
        self.exp_name = exp_name
        self.target_model_name = target_model_name

        self.exp_path = os.path.join(os.path.dirname(__file__), '..', 'models', exp_name)
        self.output_path = os.path.join(self.exp_path, 'outputs')
        image_path = os.path.join(self.exp_path, 'images')
        if os.path.exists(image_path):
            os.system(f'rm -rf {image_path}/*')

        self.sim_time = sim_time

    def run(self):
        self.merge_plt_files()
        self.visualize()

    def visualize(self):
        plt_path = os.path.join(self.exp_path, 'plts')
        plt_files = [file for file in os.listdir(plt_path) if os.path.isfile(os.path.join(plt_path, file))]
        Logger.info(f'available plots: {plt_files}')

        plt_paths = [os.path.join(plt_path, file) for file in plt_files]
        image_path = os.path.join(self.exp_path, 'images')

        if not os.path.exists(image_path):
            os.mkdir(image_path)

        plt.style.use('science')
        plt.rcParams['text.usetex'] = False

        for i, (plt_path, plt_file) in enumerate(zip(plt_paths, plt_files)):
            pdf_output_path = os.path.join(image_path, plt_file.replace('.plt', '.pdf'))
            png_output_path = os.path.join(image_path, plt_file.replace('.plt', '.png'))
            print(plt_path)

            data_list, xlabel_str, ylabel_str = load_plt_data(plt_path)

            fig, axes = plt.subplots(1, 1, figsize=(10, 4))
            min_val = 1e9

            Logger.info(f'[{i + 1}/{len(plt_paths)}] Start plotting figure {pdf_output_path}')
            for j, data in enumerate(data_list):
                try:
                    data = [list(map(float, d.split(' '))) for d in data]
                    data = np.array(data, dtype=np.float32)
                except:
                    print(data)
                    continue

                if len(data) != 5:
                    continue

                x_min = np.min(data[:, 0])
                x_max = np.max(data[:, 0])
                y_min = np.min(data[:, 1])
                y_max = np.max(data[:, 1])

                if xlabel_str == 't':
                    x_max = min(x_max, self.sim_time)
                    x_min = min(x_min, self.sim_time)

                if j == 0:
                    axes.fill_between([x_min, x_max], y_min, y_max, color='blue', label='reachable set', rasterized=True)
                else:
                    axes.fill_between([x_min, x_max], y_min, y_max, color='blue', rasterized=True)

                min_val = min(min_val, np.min(data[:, 1]))

            axes.set_xlabel(xlabel_str)
            axes.set_ylabel(ylabel_str)
            axes.legend()
            plt.tight_layout()

            plt.savefig(pdf_output_path, format='pdf')
            plt.savefig(png_output_path, format='png')

            Logger.info(f'[{i + 1}/{len(plt_paths)}] Finish plotting figure {pdf_output_path}')

            plt.close(fig)

    def merge_plt_files(self):
        plt_path = os.path.join(self.exp_path, 'plts')
        if not os.path.exists(plt_path):
            os.mkdir(plt_path)
        else:
            os.system(f'rm -rf {plt_path}/*')

        files = [file for file in os.listdir(self.output_path) if os.path.isfile(os.path.join(self.output_path, file)) and file.endswith('.plt')]
        for file in files:
            os.system(f'cp -r {os.path.join(self.output_path, file)} {os.path.join(plt_path, file)}')

        merged_files = {}
        for file in files:
            cls = '-'.join(file.replace('.plt', '').split('-')[1:])
            file_path = os.path.join(self.output_path, file)
            if cls not in merged_files:
                merged_files[cls] = [file_path]
            else:
                merged_files[cls].append(file_path)

        for key, file_paths in tqdm(merged_files.items()):
            _, xlabel_str, ylabel_str = key.split('-')

            plt_file_name = key + '.plt'
            plt_file_path = os.path.join(plt_path, plt_file_name)
            output_str = ''

            for file_path in file_paths:
                output_str += '\n' + open(file_path).read()

            open(plt_file_path, 'w', encoding='utf-8').write(output_str)


class UnsafeSetGenerator:
    def __init__(self, exp_name: str, target_model_name=None, sim_time=4.0, sim_step=1e-2):
        super(UnsafeSetGenerator).__init__()
        self.exp_name = exp_name
        self.target_model_name = target_model_name

        self.exp_path = os.path.join(os.path.dirname(__file__), '..', 'models', exp_name)
        self.plt_path = os.path.join(self.exp_path, 'plts')

        self.json_path = os.path.join(self.exp_path, 'jsons')
        if not os.path.exists(self.json_path):
            os.mkdir(self.json_path)

        self.sim_time = sim_time
        self.sim_step = sim_step

    def run(self):
        self.plt_to_set()
        self.dif_unsafe_set()

    def dif_unsafe_set(self):
        json_file_paths = [file for file in os.listdir(self.json_path) if
                           os.path.isfile(os.path.join(self.json_path, file)) and file.endswith('.json')]
        json_file_paths = [os.path.join(self.json_path, file) for file in json_file_paths]

        source_model_name = 'abnormal'
        target_model_name = 'normal'

        source_json_files = [file for file in json_file_paths if file.split('/')[-1].startswith(source_model_name)]
        target_json_files = [file for file in json_file_paths if file.split('/')[-1].startswith(target_model_name)]
        source_json_files = [file for file in source_json_files if
                             file.replace(source_model_name, target_model_name) in target_json_files]

        source_json_files.sort()
        target_json_files.sort()

        if len(source_json_files) == 0 or len(target_json_files) == 0:
            return

        for i, (source_json_file, target_json_file) in enumerate(zip(source_json_files, target_json_files)):
            source_set = ReachableSet([], [], set_file_path=source_json_file)
            target_set = ReachableSet([], [], set_file_path=target_json_file)
            print(f'source json file {len(source_set.times)}, {source_json_file}, target json file {len(target_set.times)}, {target_json_file}')
            model_name, x_var, y_var = source_json_file.split('/')[-1].replace('.json', '').split('-')

            output_path = os.path.join('/'.join(source_json_file.split('/')[:-1]),
                                       f'{x_var}-{y_var}-diff.json')

            try:
                diff_set = source_set - target_set
                unsafe_set = self.get_unsafe_set(len(diff_set.times))
                diff_set = diff_set.intersect(unsafe_set)

                diff_set.ylabel = y_var
                diff_set.xlabel = x_var
                diff_set.save(output_path)

            except:
                Logger.error(f'failed {output_path}')

    def get_unsafe_set(self, num_times):
        unsafe_segments, times = [], []

        for i in range(num_times):
            unsafe_segments.append([[0.0, 3.0]])
            times.append(i * self.sim_step)

        return ReachableSet(times, unsafe_segments)

    def plt_to_set(self):
        plt_file_paths = [os.path.join(self.plt_path, file) for file in os.listdir(self.plt_path) if
                          os.path.isfile(os.path.join(self.plt_path, file))]

        if self.target_model_name is not None:
            plt_file_paths = [plt_file_path for plt_file_path in plt_file_paths if self.target_model_name in plt_file_path]

        Logger.info(f'available plots: {plt_file_paths}')

        for i, plt_file_path in enumerate(plt_file_paths):

            output_json_path = plt_file_path.replace('.plt', '.json')
            output_json_path = output_json_path.replace('plts', 'jsons')

            data_list, xlabel_str, ylabel_str = load_plt_data(plt_file_path)

            max_sim_index = int(self.sim_time / self.sim_step + 0.5)
            n = int(self.sim_time / self.sim_step + 1 + 0.5)
            segments = []

            Logger.info(f'[{i + 1}/{len(plt_file_paths)}] Start generating {output_json_path} ...')
            for j, data in enumerate(data_list):
                try:
                    data = [list(map(float, d.split(' '))) for d in data]
                    data = np.array(data)
                except:
                    print(data)
                    continue

                if len(data) != 5:
                    continue

                x = data[:, 0]
                y = data[:, 1]

                x_min = np.min(x)
                x_max = np.max(x)
                y_min = np.min(y)
                y_max = np.max(y)

                x_min_index = int(round(x_min, 2) / self.sim_step + 0.5)
                x_max_index = int(round(x_max, 2) / self.sim_step + 0.5)

                if x_min_index > max_sim_index:
                    continue

                while len(segments) <= x_min_index:
                    segments.append([])

                if x_max_index > max_sim_index:
                    x_max_index = max_sim_index

                while len(segments) <= x_max_index:
                    segments.append([])

                for index in range(x_min_index, x_max_index + 1):
                    segments[index].append([y_min, y_max])

            for j in range(len(segments)):
                temp_segments = deepcopy(segments[j])
                temp_segments.sort()
                segments[j].clear()

                prev_segment = None
                for segment in temp_segments:
                    if prev_segment is None:
                        prev_segment = segment
                    else:
                        a = max(prev_segment[0], segment[0])
                        b = min(prev_segment[1], segment[1])
                        if a <= b:
                            prev_segment = [min(prev_segment[0], segment[0]), max(prev_segment[1], segment[1])]
                        else:
                            segments[j].append(prev_segment)
                            prev_segment = segment

                if prev_segment is not None:
                    segments[j].append(prev_segment)

            output_str = []
            for j, segment in enumerate(segments):
                line_str = [','.join(map(str, [a, b])) for a, b in segment]
                line_str = ';'.join(line_str)
                output_str.append(line_str)

            output_str = '\n'.join(output_str)

            output_set = {
                'xlabel': xlabel_str,
                'ylabel': ylabel_str,
                'sim_step': self.sim_step,
                'segments': output_str
            }

            Logger.info(f'[{i + 1}/{len(plt_file_paths)}] Finish generating {output_json_path}')
            json.dump(output_set, open(output_json_path, 'w', encoding='utf-8'), indent=4)


if __name__ == '__main__':
    exp_name = 'In2-dr'

    sim_time = 2.0
    sim_step = 0.005

    runner = FlowstarExperimentRunner(exp_name=exp_name)
    runner.run()

    visualizer = FlowstarExperimentVisualizer(exp_name=exp_name, sim_time=sim_time)
    visualizer.run()

    generator = UnsafeSetGenerator(exp_name=exp_name, sim_time=sim_time, sim_step=sim_step)
    generator.run()
