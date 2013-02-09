#!/usr/bin/python

from itertools import *
import sys
import re
import time
from random import randint, choice, shuffle
from copy import deepcopy
from collections import namedtuple

class TicTacToeException(Exception):
	pass

class Board:
	"""
	This class represents a tic-tac-toe board.
	"""

	def __init__(self, dimension, hpad, vpad):
		self.dimension = dimension

		self.hpad = hpad
		self.vpad = vpad
		self.pre_width = self._min_width(self.dimension)

		self.clear()

	def clear(self):
		"""
		Reset the tic-tac-toe board.
		"""
		self.board = [[None] * self.dimension for i in range(self.dimension)]
		self.last_moves = []

	def _print_board_line(self, before, sep, things):
		if before is None:
			before = ' ' * self.pre_width
		else:
			before = before.center(self.pre_width)

		print(before + sep + sep.join(things) + sep)

	def _things(self, things, widths):
		for i, val in enumerate(things):
			width = widths[i]

			if val is None:
				yield ' ' * width
			else:
				yield str(val).center(width)

	def _min_width(self, val):
		return self.hpad + len(str(val)) + self.hpad

	def _print_board_row(self, before, sep, things, widths):
		def pad():
			for i in range(self.vpad):
				self._print_board_line(None, sep, self._things([None] * self.dimension, widths))

		pad()
		self._print_board_line(before, sep, self._things(things, widths));
		pad()

	def print(self, symbols):
		"""
		Print a representation of the board to stdout.
		The symbols argument is a list that specifies how to
		map values in the board to symbols in the printed
		board.
		"""
		def objs_to_symbols(players):
			return (symbols[player] for player in players if player is not None)

		def places_to_symbols(places):
			for place in places:
				if place is not None:
					yield symbols[place]
				else:
					yield place

		column_widths = [max(self._min_width(val) for val in chain(objs_to_symbols(column), [i]))
		                 for i, column in enumerate(self.board, 1)]

		self._print_board_row(None, ' ', range(1, self.dimension+1), column_widths)

		for i in range(self.dimension):
			self._print_board_line(None, '+', ('-' * width for width in column_widths))
			self._print_board_row(str(i+1), '|', places_to_symbols(column[i] for column in self.board), column_widths)

		self._print_board_line(None, '+', ('-' * width for width in column_widths))

	def _put_val(self, x, y, val):
		if x >= self.dimension or y >= self.dimension:
			raise TicTacToeException('Those values are off the board!')

		self.board[x][y] = val

	def put(self, x, y, val):
		"""
		Put a value into the board. This is used for making a move.
		"""
		if self.board[x][y] is not None:
			raise TicTacToeException('This space is already occupied!')

		self._put_val(x, y, val)
		self.last_moves.append((x, y))

	def undo(self):
		"""
		Undo the last move. All moves are kept track of, so moves
		can be un-done until the board is in its initial state.
		"""
		if len(self.last_moves) == 0:
			raise TicTacToeException('No more moves to undo!')

		x, y = self.last_moves.pop()
		self._put_val(x, y, None)

	def row(self, num):
		"""
		Get a row from the board as a list.
		"""
		return (self.board[i][num] for i in range(self.dimension))

	def col(self, num):
		"""
		Get a column from the board as a list.
		"""
		return self.board[num]

	def _diag(self, a=1, b=0):
		return (self.board[i][a*i + b] for i in range(self.dimension))

	def main_diag(self):
		"""
		Get the main diagonal (from top left to bottom right)
		as a list.
		"""
		return self._diag()

	def anti_diag(self):
		"""
		Get the anti-diagonal (from bottom left to top right)
		as a list.
		"""
		return self._diag(-1, -1)

	def get_valid_moves(self):
		"""
		Get all moves that can currently be made on the board.
		"""
		return ((x, y) for x in range(self.dimension)
		               for y in range(self.dimension)
		               if self.board[x][y] is None)

	def rotate(self):
		"""
		Rotate the board counter-clockwise, the equivalent of 90 degrees.
		In other words, the top left position is now the top right position.
		"""
		self.board = [list(self.row(self.dimension - i - 1)) for i in range(self.dimension)]

	def reflect_horiz(self):
		"""
		Reflect the board from left to right. In other words, the left middle
		position is not at the right middle position and vice-versa.
		"""
		self.board.reverse()

	def reflect_vert(self):
		"""
		Reflect the board from top to bottom. In other words, the top middle
		position is now at the bottom middle position and vice-versa.
		"""
		self.board = [col.reverse() for col in self.board]

	def __eq__(self, other):
		return (self.dimension == other.dimension and
		        self.board == other.board)

	def _get_with_move_and_transform(self, point, move, move_val,
	                                       x_trans=lambda x, y: x,
	                                       y_trans=lambda x, y: y):
		x, y = point

		# First, transform get coords to real coordinates
		real_x = x_trans(x, y)
		real_y = y_trans(x, y)

		if move is not None and (real_x, real_y) == move:
			if self.board[real_x][real_y] is not None:
				raise TicTacToeException('Can not move there!')

			return move_val
		else:
			return self.board[real_x][real_y]

	def _moves_equal_with_transform(self, move,
	                                trans_move,
	                                x_trans, y_trans):
		for x in range(self.dimension):
			for y in range(self.dimension):
				if self._get_with_move_and_transform((x, y), move, -1) != \
				   self._get_with_move_and_transform((x, y), trans_move, -1, x_trans, y_trans):
					return False

		return True

	def _moves_equal_with_rotations(self, move1, move2, x_trans=(1, 0)):
		x_coeff, x_add = x_trans

		x_lambdas = [
			lambda x, y: x_coeff * x + x_add,
			lambda x, y: x_coeff * (self.dimension - y - 1) + x_add,
			lambda x, y: x_coeff * (self.dimension - x - 1) + x_add,
			lambda x, y: x_coeff * y + x_add,
		]

		y_lambdas = [
			lambda x, y: y,
			lambda x, y: x,
			lambda x, y: self.dimension - y - 1,
			lambda x, y: self.dimension - x - 1
		]

		for lx, ly in zip(x_lambdas, y_lambdas):
			if self._moves_equal_with_transform(move1, move2,
			                                    lx, ly):
				return True

		return False

	def moves_equal(self, move1, move2):
		"""
		Check whether the two given moves are equivalent with
		respect to this board, taking into account reflection and
		rotation equivalencies.
		"""
		return self._moves_equal_with_rotations(move1, move2) or \
		       self._moves_equal_with_rotations(move1, move2, (-1, self.dimension - 1))

	def is_equivalent(self, other):
		"""
		Check whether this board is equivalent to another board, taking into account
		reflection and rotation equivalencies.
		"""
		result = False

		for i in range(4):
			if self == other:
				result = True

			other.rotate()

		if result is False:
			other.reflect_horiz()

			for i in range(4):
				if self == other:
					result = True

				other.rotate()

			other.reflect_horiz()

		return result

	def get_equivalent_moves(self):
		"""
		Get a list of of unique moves, taking into account
		rotations and reflections. This will be a subset
		of the moves returned by get_valid_moves.
		"""
		possible_moves = self.get_valid_moves()
		equivalent_moves = []

		for px, py in possible_moves:
			for ex, ey in equivalent_moves:
				if self.moves_equal((px, py), (ex, ey)):
					break
			else:
				equivalent_moves.append((px, py))

		return equivalent_moves

GraphNode = namedtuple('GraphNode', 'scores dist moves')

class Game:
	symbols = ['X', 'O', 'Y', 'Z']

	def _gen_moves_graph(self, board=None, player=0):
		if board is None:
			board = Board(self.board.dimension, 0, 0)

		moves = board.get_equivalent_moves()
		players = len(self.players)

		best_move = None
		best_move_dist = 0

		graphdict = {}
		for x, y in moves:
			board.put(x, y, player)

			subnode = None
			if self._check_cats_game(board):
				scores = [1] * players
				subnode = GraphNode(dist=0,
				                    scores=scores,
				                    moves={})

			if subnode is None:
				winner = self._check_win(x, y, board)
				if winner is not None:
					scores = [0] * players
					scores[player] = 2
					subnode = GraphNode(dist=0,
					                    scores=scores,
					                    moves={})

			if subnode is None:
				subnode = self._gen_moves_graph(board, (player + 1) % players)

			if best_move is None or best_move[player] < subnode.scores[player]:
				best_move = subnode.scores
				best_move_dist = subnode.dist + 1

			graphdict[(x, y)] = subnode

			board.undo()

		return GraphNode(dist=best_move_dist,
		                 scores=best_move,
		                 moves=graphdict)

	def shuffle_players(self):
		"""
		Make the players player in a random order.
		"""
		shuffle(self.players)

	def __init__(self, *players, dimension=3, hpad=2, vpad=1):
		if len(players) > len(self.symbols):
			raise TicTacToeException('Too many players!')

		self.players = list(players)
		self.board = Board(dimension, hpad, vpad)

		self.graph = self._gen_moves_graph()
		self.graph_board = Board(dimension, 0, 0)

		self._clear_state()

	def _clear_state(self):
		self.board.clear()
		self.cur_player = None
		self.last_move = None

		self.cur_graph = self.graph
		self.graph_board.clear()

	def run(self):
		"""
		Run the game.
		"""
		print('The initial state of the board is:')
		self.board.print(self.symbols)
		print()

		for i in cycle(range(len(self.players))):
			self.cur_player = i
			print('It is {0}\'s turn!'.format(self.symbols[i]))
			print()

			self.players[i].get_move(self)

			if self.last_move is None:
				raise TicTacToeException('No move was made???')

			x, y = self.last_move
			self.last_move = None

			print('The state of the board is:')
			self.board.print(self.symbols)
			print()

			winner = self._check_win(x, y)

			if winner is not None:
				print('We have a winner! It is {0}! Congratulations!'.format(self.symbols[winner]))
				break

			if self._check_cats_game():
				print('Looks like nobody can win now - it\'s a tie! \'round these parts, we call that a "cat\'s game".')
				break

			for (x, y), node in self.cur_graph.moves.items():
				self.graph_board.put(x, y, self.cur_player)

				if self.board.is_equivalent(self.graph_board):
					self.cur_graph = node
					break

				self.graph_board.undo()

		self._clear_state()

	def _check_cats_game_sequence(self, seq):
		grouped = list(groupby(val for val in seq
		                           if val is not None))

		return len(grouped) > 1

	def _check_cats_game(self, board=None):
		if board is None:
			board = self.board

		all_seqs = chain(
			(board.row(i) for i in range(board.dimension)),
			(board.col(i) for i in range(board.dimension)),
			[board.main_diag(), board.anti_diag()]
		)

		return all(self._check_cats_game_sequence(seq) for seq in all_seqs)

	def _check_win_sequence(self, seq):
		grouped = [k for k, g in groupby(seq)]

		if len(grouped) > 1 or grouped[0] is None:
			return None
		else:
			return grouped[0]

	def _check_win(self, x, y, board=None):
		if board is None:
			board = self.board

		to_check = [board.col(x), board.row(y)]

		# main diagonal
		if x == y:
			to_check += [board.main_diag()]

		# anti-diagonal
		if x == (board.dimension - 1) - y:
			to_check += [board.anti_diag()]

		for sequence in to_check:
			player = self._check_win_sequence(sequence)

			if player is not None:
				return player

		return None

	def do_move(self, x, y):
		"""
		Do a move. This function is called by the Player classes
		in their get_move functions.
		"""
		if self.last_move is not None:
			raise TicTacToeException('Come on man, you obviously can\'t go twice...')

		self.board.put(x, y, self.cur_player)
		self.last_move = x, y

class ManualPlayer:
	def __init__(self):
		self.input_re = re.compile('^[^\d]*(\d+)[^\d]+(\d+)[^\d]*$')

	def get_move(self, game):
		while True:
			move = input('Enter your move (first is x, second is y): ')

			match = self.input_re.match(move)
			if match is None:
				print('Please try again...')
				continue

			x, y = (int(coord) for coord in match.group(1, 2))

			if x == 0 or y == 0:
				print('Both values must be more than 0!')
				continue

			try:
				game.do_move(x - 1, y - 1)
			except TicTacToeException as e:
				print(e.args[0])
				continue
			else:
				break

		print()

class AutomaticPlayer:
	def __init__(self, noises=['Beep', 'Boop']):
		self.noises = noises

	def get_move(self, game):
		for noise in islice(cycle(self.noises), randint(3, 6)):
			sys.stdout.write(noise + '... ')
			sys.stdout.flush()
			time.sleep(0.1)

		print()
		print()

		node = game.cur_graph

		# The best moves we can make have this score
		best_score = max(move.scores[game.cur_player] for move in node.moves.values())

		if best_score == 2:
			print('Ha Ha Ha I can beat you now!')
			time.sleep(0.5)
			print()

		# Among the moves that get us that score, what is the best distance to endgame?
		best_dist = min(move.dist for move in node.moves.values()
		                          if move.scores[game.cur_player] == best_score)

		best_moves = []
		for point, move_node in node.moves.items():
			if move_node.scores[game.cur_player] == best_score and \
			   move_node.dist == best_dist:
				best_moves.append(point)

		ex, ey = choice(best_moves)
		game.graph_board.put(ex, ey, game.cur_player)

		equiv_moves = []
		for x, y in game.board.get_valid_moves():
			game.board.put(x, y, game.cur_player)

			if game.board.is_equivalent(game.graph_board):
				equiv_moves.append((x, y))

			game.board.undo()

		game.graph_board.undo()

		x, y = choice(equiv_moves)

		# Let the exceptions go unhandled - nothing we can really do about them
		game.do_move(x, y)

def main():
	player1 = ManualPlayer()
	player2 = AutomaticPlayer()

	game = Game(player1, player2)

	while True:
		game.shuffle_players()
		game.run()

		prompt = input('Play again (say "yes" for affirmative)? ').lower()
		if not 'yes'.startswith(prompt):
			print('Fine. Bye.')
			break;

		print()

if __name__ == '__main__':
	main()
