from collections import defaultdict, namedtuple
import argparse
import itertools
import heapq
import math
import re

SearchNode = namedtuple('SearchNode', 'state parent action cost')

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
			plan.insert(0, node.action)
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

		for next_state, action in next_states(node.state, dest):
			child = SearchNode(next_state, node, action, node.cost + 1)

			if child.state not in explored and \
			   not frontier.has_state(child.state):
				frontier.add_node(child)
			elif frontier.has_state(child.state) and \
			     frontier.prio_of_state(child.state) > child.cost:
				frontier.add_node(child)

Coord = namedtuple('Coord', 'x y')

WumpusWorldRoomT = namedtuple('WumpusWorldRoom',
                              'pit breeze wumpus stench gold')
def WumpusWorldRoom(pit=None, breeze=None, wumpus=None, stench=None, gold=None):
	return WumpusWorldRoomT(pit, breeze, wumpus, stench, gold)

def enum(name, *vals):
	vals = [val.lower() for val in vals]

	def decorate(cls):
		for val in vals:
			setattr(cls, val.upper(), val)

		setattr(cls, name, vals)

		return cls

	return decorate

# Directions the agent is facing
@enum('directions', 'UP', 'RIGHT', 'DOWN', 'LEFT')
# Actions the agent can take
@enum('actions', 'FORWARD', 'TURN_LEFT', 'TURN_RIGHT',
                 'GRAB', 'SHOOT', 'CLIMB')
# Things that the agent can perceive
@enum('percepts', 'STENCH', 'BREEZE', 'GLITTER',
                  'BUMP', 'SCREAM')
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
				raise RuntimeException

			if normalize:
				x -= 1
				y -= 1

			val = Coord(x, y)

			if singular:
				d[key] = val
			else:
				d[key].append(val)

		return d

	def _is_valid(self, coord):
		return coord.x >= 0 and coord.y >= 0 and \
		       coord.x < self.size.x and coord.y < self.size.y

	def _next_pos(self, pos, direction):
		next_pos = {
			self.RIGHT: lambda: Coord(pos.x + 1, pos.y),
			self.LEFT:  lambda: Coord(pos.x - 1, pos.y),
			self.UP:    lambda: Coord(pos.x, pos.y + 1),
			self.DOWN:  lambda: Coord(pos.x, pos.y - 1)
		}[direction]()

		return next_pos if self._is_valid(next_pos) else None

	def _adjacent(self, coord):
		for direction in self.directions:
			next_pos = self._next_pos(coord, direction)
			if next_pos is not None:
				yield next_pos

	def _make_room(self, coord, description):
		def adj_to(coords):
			return any(adj in coords for adj in self._adjacent(coord))

		return WumpusWorldRoom(pit=coord in description['pit'],
		                       breeze=adj_to(description['pit']),
		                       wumpus=coord == description['wumpus'],
		                       stench=adj_to([description['wumpus']]) or \
		                              coord == description['wumpus'],
		                       gold=coord in description['gold'])

	def __init__(self, path, agent):
		with open(path) as f:
			description = self._parse_map_file(f)

			self.size = description['size']
			self.goal = description['goal']
			self.start = description['start']
			self.golds = len(description['gold'])

		self.world = [[self._make_room(Coord(x, y), description)
			           for y in range(self.size.y)]
		              for x in range(self.size.x)]

		self.agent = agent

	def _rel_to_start(self, coord):
		return Coord(coord.x - self.start.x, coord.y - self.start.y)

	def _percepts_at_room(self, room):
		percept_map = {
			'breeze': self.BREEZE,
			'stench': self.STENCH,
			'gold': self.GLITTER
		}

		return [v for k, v in percept_map.items() if getattr(room, k)]

	def _prev_in_list(self, elem, things):
		idx = things.index(elem) - 1

		if idx < 0:
			idx = len(things) - 1

		return things[idx]

	def _next_in_list(self, elem, things):
		idx = things.index(elem) + 1

		if idx >= len(things):
			idx = 0

		return things[idx]

	def _remove_state(self, coord, state):
		self.world[coord.x][coord.y] = \
			self.world[coord.x][coord.y]._replace(**{state: False})

	def run(self):
		direction = self.RIGHT
		score = 0
		old_pos = None
		pos = self.start
		dead = []
		arrows = 1

		agent.start(self._rel_to_start(self.goal), self.golds, direction)

		while True:
			if old_pos != pos:
				room = self.world[pos.x][pos.y]

				if room.wumpus or room.pit:
					dead = True
					break

				agent_pos = self._rel_to_start(pos)
				agent.perceive(agent_pos, self._percepts_at_room(room))
				old_pos = pos

			action = agent.get_action()

			print('Action: ' + str(action))

			if action == self.FORWARD:
				new_pos = self._next_pos(pos, direction)
				if new_pos is None:
					agent.perceive(agent_pos, [self.BUMP])
				else:
					score -= 1
					pos = new_pos
			elif action == self.TURN_LEFT:
				direction = self._prev_in_list(direction, self.directions)
			elif action == self.TURN_RIGHT:
				direction = self._next_in_list(direction, self.directions)
			elif action == self.GRAB:
				if room.gold:
					score += 1000
					self._remove_state(pos, 'gold')
			elif action == self.CLIMB:
				if pos == self.goal:
					break
			elif action == self.SHOOT:
				if arrows > 0:
					check = pos
					while True:
						check = self._next_pos(check, direction)
						if check is None:
							break

						if self.world[check.x][check.y].wumpus:
							agent.perceive(agent_pos, [self.SCREAM])
							self._remove_state(check, 'wumpus')
							break

					arrows -= 1
					score -= 100
			elif action in self.directions:
				# XXX
				direction = action

			print('\tPosition: ' + str(pos))
			print('\tScore: ' + str(score))

		if not dead:
			print('Congratulations, you survived the wumpus\' cave!')
		else:
			print('You died.')

		print('Your final score was ' + str(score))

def powerset_no_empty(iterable):
	"powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
	s = list(iterable)
	return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s)+1))

class WumpusWorldAgent:
	def _reset(self):
		self.rooms = defaultdict(lambda: set())
		self.bounds = {
			WumpusWorld.LEFT: None,
			WumpusWorld.RIGHT: None,
			WumpusWorld.UP: None,
			WumpusWorld.DOWN: None
		}
		self.pos = None
		self.current_plan = None

	def __init__(self):
		self.direction = None
		self.goal = None
		self.wumpus_alive = None
		self.gold_count = None

		self._reset()

	# FROM HERE ...
	def _next_pos(self, pos, direction):
		next_pos = {
			WumpusWorld.RIGHT: lambda: Coord(pos.x + 1, pos.y),
			WumpusWorld.LEFT:  lambda: Coord(pos.x - 1, pos.y),
			WumpusWorld.UP:    lambda: Coord(pos.x, pos.y + 1),
			WumpusWorld.DOWN:  lambda: Coord(pos.x, pos.y - 1)
		}[direction]()

		return next_pos if self._is_valid(next_pos) else None

	def _is_valid(self, coord):
		left = self.bounds[WumpusWorld.LEFT]
		right = self.bounds[WumpusWorld.RIGHT]
		bottom = self.bounds[WumpusWorld.DOWN]
		top = self.bounds[WumpusWorld.UP]

		return (left is None or coord.x >= left) and \
		       (right is None or coord.x <= right) and \
		       (bottom is None or coord.y >= bottom) and \
		       (top is None or coord.y <= top)

	def _adjacent(self, coord):
		for direction in WumpusWorld.directions:
			next_pos = self._next_pos(coord, direction)
			if next_pos is not None:
				yield next_pos
	# ... TO HERE IS VERY SIMILAR TO SOME CODE IN WumpusWorld!

	def perceive(self, pos, percepts):
		print('Percepts: ' + str(percepts))

		if WumpusWorld.BUMP in percepts:
			if self.direction == WumpusWorld.LEFT or \
			   self.direction == WumpusWorld.RIGHT:
				self.bounds[self.direction] = pos.x
			else:
				self.bounds[self.direction] = pos.y

			percepts.remove(WumpusWorld.BUMP)

			# Our current strategy obviously isn't
			# going to work
			self.current_plan = None

		if WumpusWorld.SCREAM in percepts:
			self.wumpus_alive = False
			percepts.remove(WumpusWorld.SCREAM)

		self.rooms[pos] |= set(percepts)
		self.pos = pos

	def start(self, goal, golds, direction):
		self._reset()

		self.direction = direction
		self.goal = goal
		self.wumpus_alive = True
		self.gold_count = golds

	def _get_border_rooms(self):
		rooms = frozenset()

		for coord in self.rooms.keys():
			rooms = rooms | frozenset(adj for adj in self._adjacent(coord)
			                              if adj not in self.rooms)

		return rooms

	def _safe(self, coord, thing):
		percept = {
			'wumpus': WumpusWorld.STENCH,
			'pit': WumpusWorld.BREEZE
		}[thing]

		return coord in self.rooms or \
		       any(adj in self.rooms and percept not in self.rooms[adj]
		           for adj in self._adjacent(coord))

	def _unsafe_adjacent(self, coord, thing):
		for adj in self._adjacent(coord):
			if not self._safe(adj, thing):
				yield adj

	def _detect_wumpus(self):
		stenches = (coord for coord, percepts in self.rooms.items()
		                  if WumpusWorld.STENCH in percepts)

		models = None

		for coord in stenches:
			potential_models = frozenset(frozenset([adj]) for adj in self._unsafe_adjacent(coord, 'wumpus'))

			if models is None:
				models = potential_models
			else:
				product = itertools.product(models, potential_models)
				models = frozenset(first & second for first, second in product)
				models -= frozenset()

		if models is None:
			models = frozenset([frozenset()])

		return models

	def _detect_pits(self):
		breezes = (coord for coord, percepts in self.rooms.items()
		                 if WumpusWorld.BREEZE in percepts)

		models = None

		for coord in breezes:
			potential_models = frozenset(frozenset(coords) for coords in powerset_no_empty(self._unsafe_adjacent(coord, 'pit')))

			if models is None:
				models = potential_models
			else:
				product = itertools.product(models, potential_models)
				models = frozenset(first | second for first, second in product)
				models -= frozenset()

		if models is None:
			models = frozenset([frozenset()])

		return models

	def _next_states(self, state, dest):
		for direction in WumpusWorld.directions:
			next_state = self._next_pos(state, direction)
			if next_state in self.rooms or next_state == dest:
				yield next_state, direction

	def _plan_movement(self, dest):
		return plan_movement(self.pos, dest, self._next_states)

	def _explore(self):
		def dist(coord):
			return math.sqrt((coord.x - self.pos.x)**2 + (coord.y - self.pos.y)**2)

		borders = sorted(self._get_border_rooms(), key=dist)
		wumpuses = self._detect_wumpus()
		pits = self._detect_pits()

		for wumpus_model, pit_model in itertools.product(wumpuses, pits):
			print('wumpus: ' + str(wumpus_model))
			print('pit: ' + str(pit_model))
			self._visualize_knowledge(wumpus_model, pit_model);
			print()

		print('border rooms: ' + str(borders))

		for room in borders:
			# skip over rooms that might be unsafe
			if any(room in model for model in wumpuses) or \
			   any(room in model for model in pits):
				continue

			self.current_plan = self._plan_movement(room)
			return

		for room in borders:
			# skip over rooms that are definitely unsafe
			if all(room in model for model in wumpuses) or \
			   all(room in model for model in pits):
				continue

			self.current_plan = self._plan_movement(room)
			return

		# nowhere is safe - bail out
		self.current_plan = self._plan_movement(self.goal)
		self.current_plan.append(WumpusWorld.CLIMB)


	def _room_string(self, coord, wumpus_model, pit_model):
		things = []

		if coord == self.goal:
			things.append('GO')

		if coord in self.rooms:
			percept_thing_map = {
				WumpusWorld.BREEZE: 'B',
				WumpusWorld.STENCH: 'S',
				WumpusWorld.GLITTER: 'G'
			}

			for p in self.rooms[coord]:
				things.append(percept_thing_map[p])

			things.append('E')

		if coord in wumpus_model:
			things.append('W')

		if coord in pit_model:
			things.append('P')

		if coord == self.pos:
			things.append('A')

		return ','.join(things)

	def _visualize_knowledge(self, wumpus_model, pit_model):
		known_coords = set(self.rooms.keys()) | \
		               wumpus_model | \
		               pit_model | \
		               {self.goal}

		lowest_x = self.bounds[WumpusWorld.LEFT] or min(x for x, y in known_coords)
		highest_x = self.bounds[WumpusWorld.RIGHT] or max(x for x, y in known_coords)

		lowest_y = self.bounds[WumpusWorld.DOWN] or min(y for x, y in known_coords)
		highest_y = self.bounds[WumpusWorld.UP] or max(y for x, y in known_coords)

		pre_space = 3 if self.bounds[WumpusWorld.LEFT] is None else 0
		post_dots = 3 if self.bounds[WumpusWorld.RIGHT] is None else 0

		grid = [[self._room_string(Coord(x, y), wumpus_model, pit_model) for y in range(lowest_y, highest_y+1)]
		        for x in range(lowest_x, highest_x+1)]

		column_widths = [max(max(len(s), 3) for s in column) for column in grid]
		column_dashes = ['-' * width for width in column_widths]

		dots_line = '.'.join(' ' * num for num in [pre_space] + column_widths) + '.'
		pre_dots = '.' * pre_space
		post_dots = '.' * post_dots

		between_line = '+'.join([pre_dots] + column_dashes + [post_dots])

		def dots():
			for _ in range(3): print(dots_line)

		if self.bounds[WumpusWorld.DOWN] is None:
			dots()

		for x, colnum in enumerate(range(lowest_y, highest_y+1)):
			print(between_line)
			print(' ' * pre_space + '|' + '|'.join(col[x].ljust(column_widths[y]) for y, col in enumerate(grid)) + '|')

		print(between_line)

		if self.bounds[WumpusWorld.UP] is None:
			dots()

	def get_action(self):
		room = self.rooms[self.pos]

		if 'glitter' in room:
			room.remove('glitter')
			self.gold_count -= 1
			return WumpusWorld.GRAB

		if not self.current_plan:
			if self.gold_count == 0:
				self.current_plan = self._plan_movement(self.goal)
				self.current_plan.append(WumpusWorld.CLIMB)
			else:
				self._explore()

		action = self.current_plan[0]
		if action in WumpusWorld.directions:
			if self.direction != action:
				self.direction = action
				return action
			else:
				self.current_plan.pop(0)
				return WumpusWorld.FORWARD
		else:
			self.current_plan.pop(0)
			return action

parser = argparse.ArgumentParser(description='Wumpus World with AI')
parser.add_argument('-w', '--world-file',
                    default='wumpus_world.txt',
                    metavar='/path/to/map/file')
args = parser.parse_args()

agent = WumpusWorldAgent()
game = WumpusWorld(args.world_file, agent)
game.run()
