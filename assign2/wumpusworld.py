from collections import defaultdict, namedtuple
from functools import reduce
import argparse
import itertools
import heapq
import math
import re

# A modified version of the standard defaultdict which assumes
# that the data being stored in the dict will be immutable.
# We can make some extra assumptions - for example, we do not need
# a factory for the default values, only a value. Also, when
# getting a default value, we do not have to add it to the dict
# just yet.
class immutabledefaultdict(dict):
	def __init__(self, default_val=None, *a, **kw):
		dict.__init__(self, *a, **kw)
		self.default_val = default_val
	def __getitem__(self, key):
		try:
			return dict.__getitem__(self, key)
		except KeyError:
			return self.__missing__(key)
	def __missing__(self, key):
		if self.default_val is None:
			raise KeyError(key)
		return self.default_val
	def __reduce__(self):
		if self.default_val is None:
			args = tuple()
		else:
			args = self.default_val,
		return type(self), args, None, None, self.items()
	def copy(self):
		return self.__copy__()
	def __copy__(self):
		return type(self)(self.default_val, self)
	def __deepcopy__(self, memo):
		import copy
		return type(self)(self.default_val,
				copy.deepcopy(self.items()))
		def __repr__(self):
			return 'defaultdict(%s, %s)' % (self.default_val,
					dict.__repr__(self))

def color(stuff, cnums):
	return '\x1b[{0}m{1}\x1b[m'.format(';'.join(str(cnum) for cnum in cnums), stuff)

SearchNode = namedtuple('SearchNode', 'state parent actions cost')

class FrontierQueue:
	REMOVED = '<removed-task>'      # placeholder for a removed task

	def __init__(self):
		self.heap = []
		self.finder = {}
		self.counter = itertools.count()

	def add_node(self, node):
		if node.state in self.finder:
			self.remove_state(node.state)
		count = next(self.counter)
		entry = (node.cost, count, node)
		self.finder[node.state] = entry
		heapq.heappush(self.heap, entry)

	def has_state(self, state):
		return state in self.finder and \
		       self.finder[state][-1] != self.REMOVED

	def prio_of_state(self, state):
		return self.finder[state][0]

	def remove_state(self, state):
		entry = self.finder.pop(state)
		entry[-1] = self.REMOVED

	def pop_node(self):
		while self.heap:
			priority, count, node = heapq.heappop(self.heap)
			if node is not self.REMOVED:
				del self.finder[node.state]
				return node
		raise KeyError('pop from an empty frontier queue')

# The search algorithm
def plan_movement(start, dest, next_states):
	def solution(node):
		plan = []

		while node.parent is not None:
			plan = node.actions + plan
			node = node.parent

		return plan

	node = SearchNode(start, None, None, 0)
	frontier = FrontierQueue()
	frontier.add_node(node)
	explored = set()

	while True:
		if not frontier:
			return None

		node = frontier.pop_node()
		if node.state == dest:
			return solution(node)

		explored.add(node.state)

		for next_state, actions in next_states(node.state, dest):
			child = SearchNode(next_state, node, actions, node.cost + 1)

			if child.state not in explored and \
			   not frontier.has_state(child.state):
				frontier.add_node(child)
			elif frontier.has_state(child.state) and \
			     frontier.prio_of_state(child.state) > child.cost:
				frontier.add_node(child)

def enum(name, *vals):
	vals = [val.lower() for val in vals]

	def decorate(cls):
		for val in vals:
			setattr(cls, val.upper(), val)

		setattr(cls, name, vals)

		return cls

	return decorate

Coord = namedtuple('Coord', 'x y')

class WumpusWorldMap:
	AxisInfo = namedtuple('AxisInfo', 'low high')
	DirInfo = namedtuple('DirInfo', 'extreme axis')
	WumpusWorldRoom = namedtuple('WumpusWorldRoom',
	                             'explored pit breeze wumpus stench gold')

	axes = {
		# x increases from west to east
		'x': AxisInfo('west', 'east'),
		# y increases from south to north
		'y': AxisInfo('south', 'north')
	}

	directions = {}
	for axis, (low, high) in axes.items():
		directions[low] = DirInfo('low', axis)
		directions[high] = DirInfo('high', axis)

	def __init__(self):
		self.rooms = immutabledefaultdict(self.WumpusWorldRoom(False, None, None,
		                                                       None, None, None))
		self.bounds = {direction: None for direction in self.directions}

	def _on_board_axis(self, coord, axis, low=None, high=None):
		if low is None:
			low = self.bounds[self.axes[axis].low]

		if high is None:
			high = self.bounds[self.axes[axis].high]

		if low is not None and high is not None and low > high:
			raise RuntimeError()

		val = getattr(coord, axis)

		return (low is None or val >= low) and \
		       (high is None or val <= high)

	def on_board(self, coord):
		return all(self._on_board_axis(coord, axis) for axis in self.axes)

	def next_pos(self, pos, direction):
		extreme, axis = self.directions[direction]
		diff = 1 if extreme == 'high' else -1

		pos = pos._replace(**{axis: getattr(pos, axis) + diff})

		return pos if self.on_board(pos) else None

	def adjacent(self, coord):
		for direction in self.directions:
			next_pos = self.next_pos(coord, direction)
			if next_pos is not None:
				yield next_pos

	def set_extremes(self, axis, low=None, high=None):
		#if any(not self._on_board_axis(coord, axis, low, high)
		#       for coord in self.rooms.keys()):
		#	raise RuntimeError()
		for coord in list(self.rooms):
			if not self._on_board_axis(coord, axis, low, high):
				del self.rooms[coord]

		if low is not None:
			self.bounds[self.axes[axis].low] = low

		if high is not None:
			self.bounds[self.axes[axis].high] = high

	def set_extreme_at(self, pos, direction):
		extreme, axis = self.directions[direction]

		self.set_extremes(axis, **{extreme: getattr(pos, axis)})

	def add_knowledge(self, coord, **kwargs):
		if not self.on_board(coord):
			raise RuntimeError()

		changed = {key: val for key, val in kwargs.items()
		                    if getattr(self.rooms[coord], key) != val}

		self.rooms[coord] = self.rooms[coord]._replace(**changed)

		return changed

	def get_border_rooms(self):
		rooms = set()

		for coord, room in self.rooms.items():
			if room.explored:
				rooms |= {adj for adj in self.adjacent(coord)
				              if not self.rooms[adj].explored}

		return rooms

	def _check_vals(self, coord, op, fields, val):
		return op(getattr(self.rooms[coord], field) is val
		          for field in fields)

	def all_vals(self, coord, fields, val):
		return self._check_vals(coord, all, fields, val)

	def any_vals(self, coord, fields, val):
		return self._check_vals(coord, any, fields, val)

	def adjacent_with(self, coord, things, val):
		return (adj for adj in self.adjacent(coord)
		            if self.all_vals(adj, things, val))

	def adjacent_with_not(self, coord, things, val):
		return (adj for adj in self.adjacent(coord)
		            if not self.any_vals(adj, things, val))

	def _room_string(self, coord, field_sym_map, extras):
		things = []

		room = self.rooms[coord]

		for f in field_sym_map:
			sym, colors = field_sym_map[f]
			val = getattr(room, f) if hasattr(room, f) else None

			if f in extras and coord in extras[f]:
				if val is False:
					raise RuntimeError()

				val = True

			if val is False:
				sym = '!' + sym
			elif val is True:
				sym = color(sym, colors)

			if val is not None:
				things.append(sym)

		return ','.join(things)

	def _printed_len(self, s):
		in_escseq = False
		length = 0

		for char in s:
			if char == '\x1b':
				in_escseq = True

			if in_escseq:
				if char == 'm':
					in_escseq = False

				continue

			length += 1

		return length

	def _ljust(self, s, w):
		l = self._printed_len(s)
		if l < w:
			return s + (' ' * (w - l))
		else:
			return s

	def visualize_knowledge(self, field_sym_map, extras):
		known_coords = set(self.rooms.keys()).union(*extras.values())

		def extreme(extreme, axis):
			direction = getattr(self.axes[axis], extreme)

			if self.bounds[direction] is not None:
				return self.bounds[direction]
			else:
				op = {
					'low': min,
					'high': max
				}[extreme]

				return op(getattr(coord, axis) for coord in known_coords)

		lowest_x = extreme('low', 'x')
		highest_x = extreme('high', 'x') + 1

		lowest_y = extreme('low', 'y')
		highest_y = extreme('high', 'y') + 1

		num_dots = 3
		pre = num_dots if self.bounds['west'] is None else 0
		post = num_dots if self.bounds['east'] is None else 0

		grid = [[self._room_string(Coord(x, y), field_sym_map, extras)
		         for y in range(lowest_y, highest_y)]
		        for x in range(lowest_x, highest_x)]

		col_widths = [max(max(self._printed_len(s), 3) for s in column) for column in grid]

		dots_line = '.'.join(' ' * width for width in [pre] + col_widths) + '.'

		col_dashes = ['-' * width for width in col_widths]
		between_line = '+'.join(['.' * pre] + col_dashes + ['.' * post])

		def dots():
			for _ in range(num_dots): print(dots_line)

		if self.bounds['north'] is None:
			dots()

		for x in reversed(range(highest_y - lowest_y)):
			print(between_line)
			col_vals = [self._ljust(col[x], width) for width, col in zip(col_widths, grid)]
			print('|'.join([' ' * pre] + col_vals) + '|')

		print(between_line)

		if self.bounds['south'] is None:
			dots()

# Actions the agent can take
# The agent can also face a certain direction with the
# WumpusWorldMap.directions values.
@enum('actions', 'FORWARD', 'GRAB', 'SHOOT', 'CLIMB')
# Room-independent things that the agent can perceive
# The room-dependent things are stored in the WumpusWorldRoom
# namedtuple
@enum('percepts', 'BUMP', 'SCREAM')
class WumpusWorld:
	map_file_descriptions = {
		'M': ('size', True, False),
		'A': ('start', True, True),
		'GO': ('goal', True, True),

		'P': ('pit', False, True),

		'W': ('wumpus', True, True),

		'G': ('gold', False, True),
	}

	map_line_re = re.compile('^([A-Z]+)(\d)(\d)$')

	def _parse_map_file(self, f):
		d = defaultdict(lambda: [])

		for line in f:
			line = line.strip()

			match = self.map_line_re.match(line)
			if match is None:
				continue

			type, x, y = match.group(1, 2, 3)
			x = int(x)
			y = int(y)

			if type not in self.map_file_descriptions:
				continue

			key, singular, normalize = self.map_file_descriptions[type]

			if singular and key in d:
				raise RuntimeError()

			if normalize:
				x -= 1
				y -= 1

			val = Coord(x, y)

			if singular:
				d[key] = val
			else:
				d[key].append(val)

		return d

	def __init__(self, path, agent):
		with open(path) as f:
			description = self._parse_map_file(f)

			size = description['size']
			self.goal = description['goal']
			self.start = description['start']
			self.golds = len(description['gold'])

		self.map = WumpusWorldMap()
		self.map.set_extremes('x', 0, size.x - 1)
		self.map.set_extremes('y', 0, size.y - 1)

		def adj_to(coords):
			return any(adj in coords for adj in self.map.adjacent(coord))

		for x, y in itertools.product(range(size.x), range(size.y)):
			coord = Coord(x, y)
			self.map.add_knowledge(coord,
			                       explored=True,
			                       pit=coord in description['pit'],
			                       breeze=adj_to(description['pit']),
			                       wumpus=coord == description['wumpus'],
			                       stench=adj_to([description['wumpus']]) or \
			                              coord == description['wumpus'],
			                       gold=coord in description['gold'])

		self.agent = agent

	def _rel_to_start(self, coord):
		return Coord(coord.x - self.start.x, coord.y - self.start.y)

	def run(self):
		direction = 'east'
		score = 0
		old_pos = None
		pos = self.start
		dead = []
		arrows = 1

		agent.start(self._rel_to_start(self.goal), self.golds, direction)

		while True:
			if old_pos != pos:
				room = self.map.rooms[pos]

				if room.wumpus or room.pit:
					dead = True
					break

				agent_pos = self._rel_to_start(pos)
				agent.perceive(agent_pos, room=room)
				old_pos = pos

			action = agent.get_action()

			print('Action: ' + str(action))

			if action == self.FORWARD:
				new_pos = self.map.next_pos(pos, direction)
				if new_pos is None:
					agent.perceive(agent_pos, others=[self.BUMP])
				else:
					score -= 1
					pos = new_pos
			elif action == self.GRAB:
				if room.gold:
					score += 1000
					self.map.add_knowledge(pos, gold=False)
			elif action == self.CLIMB:
				if pos == self.goal:
					break
			elif action == self.SHOOT:
				if arrows > 0:
					check = pos
					while True:
						check = self.map.next_pos(check, direction)
						if check is None:
							break

						if self.map.rooms[check].wumpus:
							agent.perceive(agent_pos, others=[self.SCREAM])
							self.map.add_knowledge(check, wumpus=False)
							break

					arrows -= 1
					score -= 100
			elif action in WumpusWorldMap.directions:
				direction = action

			print('\tPosition: ' + str(pos))
			print('\tScore: ' + str(score))

		if not dead:
			print('Congratulations, you survived the wumpus\' cave!')
		else:
			print('You died.')

		print('Your final score was ' + str(score))

def powerset(iterable, minlen=0):
	"powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
	s = list(iterable)
	return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(minlen, len(s)+1))

class WumpusWorldAgent:
	def _reset(self):
		self.pos = None
		self.current_plan = None

	def __init__(self):
		self.direction = None
		self.goal = None
		self.screams = None
		self.gold_count = None
		self.map = None

		self._reset()

	def start(self, goal, golds, direction):
		self.direction = direction
		self.goal = goal
		self.screams = 0
		self.gold_count = golds
		self.map = WumpusWorldMap()

		self._reset()

	def perceive(self, pos, room=None, others=[]):
		self.pos = pos

		print('Percepts: ' + str(others))

		if WumpusWorld.BUMP in others:
			self.map.set_extreme_at(pos, self.direction)

			# Our current strategy obviously isn't
			# going to work
			self.current_plan = None

		if WumpusWorld.SCREAM in others:
			self.screams += 1

		if room:
			print('Room: ' + str(room))
			self._add_knowledge(pos, **room._asdict())

	DetectableInfo = namedtuple('DetectableInfo', 'percept limit')

	detectables = {
		'wumpus': DetectableInfo('stench', 1),
		'pit': DetectableInfo('breeze', float('inf'))
	}

	field_sym_map = {
		'explored': ('E', ()),
		'pit': ('P', (47, 30)),
		'breeze': ('B', (47, 30)),
		'stench': ('S', (41, 30)),
		'wumpus': ('W', (41, 30)),
		'gold': ('G', (45, 30)),
		'goal': ('GO', ()),
		'agent': ('A', ())
	}

	def _add_knowledge(self, coord, **kwargs):
		changed = self.map.add_knowledge(coord, **kwargs)

		for detectable, info in self.detectables.items():
			if info.percept in changed and changed[info.percept] is False:
				for adj in self.map.adjacent(coord):
					self._add_knowledge(adj, **{detectable: False})

	def _detect(self, thing):
		percept, limit = self.detectables[thing]

		percepts = (coord for coord, room in self.map.rooms.items()
		                  if getattr(room, percept))

		definites = set()

		models = {frozenset()}

		for coord in percepts:
			definites |= set(self.map.adjacent_with(coord, [thing], True))
			potential_models = {frozenset(coords) - definites
			                    for coords in powerset(self.map.adjacent_with_not(coord, [thing], False), 1)}
			product = itertools.product(models, potential_models)
			models = {a | b for a, b in product}

		models = {model for model in models if len(model) + len(definites) <= limit}

		return models

	def _next_states(self, state, dest):
		for direction in WumpusWorldMap.directions:
			next_state = self.map.next_pos(state, direction)
			if next_state is None:
				continue

			if self.map.all_vals(next_state, self.detectables.keys(), False) or next_state == dest:
				yield next_state, [direction, WumpusWorld.FORWARD]

	def _plan_movement(self, dest):
		return plan_movement(self.pos, dest, self._next_states)

	def _detect_definites(self, models, thing):
		if not models:
			return models

		definites = reduce(lambda a, b: a & b, models)

		for definite in definites:
			print('Deduced {0} at {1}'.format(thing, definite))
			self._add_knowledge(definite, **{thing: True})

		return {model - definites for model in models}

	def _explore(self):
		def dist(coord):
			return math.sqrt((coord.x - self.pos.x)**2 + (coord.y - self.pos.y)**2)

		detectable_models = {}
		for detectable in self.detectables:
			models = self._detect_definites(self._detect(detectable), detectable)
			detectable_models[detectable] = models

			print(detectable + ': ' + str(models))
			for model in models:
				self.map.visualize_knowledge(self.field_sym_map, {
					detectable: model,
					'agent': [self.pos],
					'goal': [self.goal]
				})

				print()

		borders = sorted(self.map.get_border_rooms(), key=dist)
		print('border rooms: ' + str(borders))

		for room in borders:
			# skip over rooms that might be unsafe
			if any(room in model for models in detectable_models.values()
			                     for model in models) or \
			   not self.map.all_vals(room, self.detectables.keys(), False):
				continue

			self.current_plan = self._plan_movement(room)
			return

		for room in borders:
			# skip over rooms that are definitely unsafe
			if self.map.any_vals(room, self.detectables.keys(), True):
				continue

			self.current_plan = self._plan_movement(room)
			return

		# nowhere is safe - bail out
		self.current_plan = self._plan_movement(self.goal)
		self.current_plan.append(WumpusWorld.CLIMB)

	def get_action(self):
		room = self.map.rooms[self.pos]

		if room.gold:
			self._add_knowledge(self.pos, gold=False)
			self.gold_count -= 1
			return WumpusWorld.GRAB

		if not self.current_plan:
			if self.gold_count == 0:
				self.current_plan = self._plan_movement(self.goal)
				self.current_plan.append(WumpusWorld.CLIMB)
			else:
				self._explore()

		while self.current_plan:
			action = self.current_plan.pop(0)

			if action in WumpusWorldMap.directions:
				if action == self.direction:
					continue

				self.direction = action

			return action

parser = argparse.ArgumentParser(description='Wumpus World with AI')
parser.add_argument('-w', '--world-file',
                    default='wumpus_world.txt',
                    metavar='/path/to/map/file')
args = parser.parse_args()

agent = WumpusWorldAgent()
game = WumpusWorld(args.world_file, agent)
game.run()
