import re
import json
import numpy as np
from copy import deepcopy


def load_plt_data(plt_path):
    data = open(plt_path, 'r', encoding='utf-8').readlines()
    data_list = []
    current_data = []
    xlabel_str, ylabel_str = '', ''

    for d in data:
        if len(d.strip()) > 0:
            if d[0] not in ['-'] + [str(i) for i in range(10)]:
                if 'set xlabel' in d:
                    xlabel_str = re.match('set xlabel "(.+)"', d).group(1)

                if 'set ylabel' in d:
                    ylabel_str = re.match('set ylabel "(.+)"', d).group(1)

            else:
                current_data.append(d)
        else:
            if len(current_data) > 0:
                data_list.append(current_data)
            current_data = []

    return data_list, xlabel_str, ylabel_str


def cross_product(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def choose_action(n):
    weights = np.zeros(n)
    weights[-1] = 1.0
    weights[-1] = 0.2

    cumulative_weights = np.cumsum(weights)

    random_number = np.random.rand() * cumulative_weights[-1]

    action = np.searchsorted(cumulative_weights, random_number)

    return action


def cross_entropy_method(env, n_iterations=100, batch_size=64, elite_frac=0.2, n_elite=20):
    n_actions = env.action_space.n
    action_prob = np.exp(np.arange(1, n_actions + 1))
    action_prob = np.array(action_prob.tolist(), dtype=np.float32)
    action_prob /= np.sum(action_prob)
    reward_history = []

    for iteration in range(n_iterations):
        print(f'action prob {action_prob}')
        batch = []
        for _ in range(batch_size):
            episode = []
            state, info = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = np.random.choice(n_actions, p=action_prob)
                if not env.is_action_valid(action):
                    action = 0

                observation, reward, terminated, truncated, info = env.step(action)

                if env.q == 3:
                    episode.append((state, action, reward))

                total_reward = reward
                state = observation
                done = terminated or truncated

            batch.append((episode, total_reward))

        batch.sort(key=lambda x: x[1], reverse=True)
        elite_episodes = [episode for episode, reward in batch[:n_elite]]

        action_counts = np.zeros(n_actions)
        for episode in elite_episodes:
            for state, action, reward in episode:
                action_counts[action] += 1
        action_prob = action_counts / action_counts.sum()
        action_prob /= np.sum(action_prob)

        reward_history.append(np.mean([reward for _, reward in batch]))

        print(f"Iteration {iteration + 1}: Mean Reward: {reward_history[-1]}")

    return action_prob, reward_history


class ReachableSet:
    def __init__(self, times, segments, set_file_path=None):
        super(ReachableSet, self).__init__()

        assert isinstance(times, list)
        assert isinstance(segments, list)

        if set_file_path is not None:
            data = json.load(open(set_file_path, 'r', encoding='utf-8'))
            self.xlabel = data['xlabel']
            self.ylabel = data['ylabel']
            self.sim_step = data['sim_step']

            segments = []

            for d in data['segments'].split('\n'):
                current_segments = []
                for item in d.strip().split(';'):
                    if len(item) > 0:
                        current_segments.append(list(map(float, item.split(','))))
                segments.append(current_segments)

            self.times = [i * self.sim_step for i in range(len(segments))]
            self.segments = segments
        else:
            self.xlabel = 't'
            self.ylabel = 'var'
            self.sim_step = times[1] - times[0]
            self.times = times

            for i in range(len(segments)):
                temp_segments = deepcopy(segments[i])
                temp_segments.sort()
                segments[i].clear()

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
                            segments[i].append(prev_segment)
                            prev_segment = segment

                if prev_segment is not None:
                    segments[i].append(prev_segment)

            self.segments = segments

        self.bound_points = np.array(self.get_bound_points())
        self.convex_hull = self.get_convex_hull()
        self.inf = 1e9

    def get_bound_points(self):
        bound_points = []

        for i in range(len(self.segments)):
            t = i * self.sim_step
            for segment in self.segments[i]:
                bound_points.append([t, segment[0]])
                bound_points.append([t, segment[1]])

        return bound_points

    def get_convex_hull(self):
        bound_points = self.get_bound_points()
        bound_points.sort()

        lower = []
        for p in bound_points:
            while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(bound_points):
            while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        return lower[:-1] + upper[:-1]

    def __contains__(self, item):
        index = int(item[0] / self.sim_step + 0.5)
        if index == -1:
            return False
        for segment in self.segments[index]:
            lower_bound, upper_bound = segment
            if lower_bound <= item[1] <= upper_bound:
                return True
        return False

    def intersect(self, other):
        assert isinstance(other, ReachableSet)
        other_step = other.times[1] - other.times[0]
        self_step = self.times[1] - self.times[0]
        if abs(other_step - self_step) > 1e-6:
            raise Exception

        segments = []
        for i in range(len(self.times)):
            segments.append([])
            for j, self_segment in enumerate(self.segments[i]):
                a, b = self_segment
                for k, other_segment in enumerate(other.segments[i]):
                    c, d = other_segment

                    e, f = max(a, c), min(b, d)
                    if e < f:
                        segments[i].append([e, f])

        return ReachableSet(self.times, segments)

    def get_num_segments(self):
        num_segments = 0
        for segment in self.segments:
            num_segments += len(segment)
        return num_segments

    def __add__(self, other):
        assert isinstance(other, ReachableSet)
        if len(other.times) != len(self.times) or len(other.times) <= 1:
            raise Exception
        other_step = other.times[1] - other.times[0]
        self_step = self.times[1] - self.times[0]
        if abs(other_step - self_step) > 1e-6:
            raise Exception

        segments = []
        for i in range(len(self.times)):
            segments.append([])
            for j, self_segment in enumerate(self.segments[i]):
                a, b = self_segment
                for k, other_segment in enumerate(other.segments[i]):
                    c, d = other_segment
                    if [a, b] not in segments[i]:
                        segments[i].append([a, b])
                    if [c, d] not in segments[i]:
                        segments[i].append([c, d])

        for i in range(len(segments)):
            temp_segments = deepcopy(segments[i])
            temp_segments.sort()
            segments[i].clear()

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
                        segments[i].append(prev_segment)
                        prev_segment = segment

            if prev_segment is not None:
                segments[i].append(prev_segment)

        return ReachableSet(self.times, segments)

    def __sub__(self, other):
        assert isinstance(other, ReachableSet)
        other_step = other.times[1] - other.times[0]
        self_step = self.times[1] - self.times[0]
        if abs(other_step - self_step) > 1e-6:
            raise Exception

        segments = []
        for i in range(len(self.times)):
            segments.append([])
            for j, self_segment in enumerate(self.segments[i]):
                a, b = self_segment
                if i >= len(other.segments):
                    segments[i].append([a, b])
                    continue

                for k, other_segment in enumerate(other.segments[i]):
                    c, d = other_segment

                    if b <= c or a >= d:
                        segments[i].append([a, b])
                    elif c <= a and d >= b:
                        pass
                    elif c > a and d < b:
                        segments[i].append([a, c])
                        segments[i].append([d, b])
                    elif c <= a < d < b:
                        segments[i].append([d, b])
                    elif a < c < b <= d:
                        segments[i].append([a, c])
                    else:
                        segments[i].append([a, b])

        return ReachableSet(self.times, segments)

    def save(self, file_path):
        output_strs = []
        for segment in self.segments:
            output_strs.append(';'.join([','.join(map(str, item)) for item in segment]))

        outputs = {"xlabel": self.xlabel, "ylabel": self.ylabel, "sim_step": self.sim_step, "segments": '\n'.join(output_strs)}
        json.dump(outputs, open(file_path, 'w', encoding='utf-8'))