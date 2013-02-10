#!/usr/bin/python

from itertools import *
import sys
import re
import time
from random import randint, choice, shuffle
from collections import namedtuple

class TicTacToeException(Exception):
	pass

# For any given tictactoe board, there are at most
# eight different orientations. If we number the corners
# one through four:
#
# 1--2    2--1
# |  |    |  |
# 4--3    3--4
#
# 4--1    1--4
# |  |    |  |
# 3--2    2--3
#
# 3--4    4--3
# |  |    |  |
# 2--1    1--2
#
# 2--3    3--2
# |  |    |  |
# 1--4    4--1
#
# For any tictactoe board, transforming between
# any of these orientations does not change the
# essential nature of the game. We use this fact for
# determining the equivalency of boards (is_equivalent),
# determining the equivalency of moves with respect
# to a board (move_equal), and for getting a list
# of the essentially unique moves that a player
# can make (get_unique_moves).
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
		for val, width in zip(things, widths):
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
		Get a row from the board as an iterable.
		"""
		return (self.board[i][num] for i in range(self.dimension))

	def col(self, num):
		"""
		Get a column from the board as an iterable.
		"""
		return self.board[num]

	def _diag(self, a=1, b=0):
		return (self.board[i][a*i + b] for i in range(self.dimension))

	def main_diag(self):
		"""
		Get the main diagonal (from top left to bottom right)
		as an iterable.
		"""
		return self._diag()

	def anti_diag(self):
		"""
		Get the anti-diagonal (from bottom left to top right)
		as an iterable.
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
		Rotate the board clockwise, the equivalent of 90 degrees.
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
		# In calling this function, we are essentially asking:
		# if I take the current board, transform our "view" of it
		# (i.e, the opposite of transforming the board itself),
		# and make a move at the specified location, what will
		# the value at the specified point be?
		x, y = point

		# This transforms the view
		real_x = x_trans(x, y)
		real_y = y_trans(x, y)

		if move is not None and (real_x, real_y) == move:
			if self.board[real_x][real_y] is not None:
				raise TicTacToeException('Can not move there!')

			# If we want to get the value at the location
			# that we are pretending to make a move, return
			# the pretend move value
			return move_val
		else:
			return self.board[real_x][real_y]

	def _moves_equal_with_transform(self, move,
	                                trans_move,
	                                x_trans, y_trans):
		# Check whether every square of this board with the given move made
		# is the same as every square of this board with the OTHER given move
		# made and with an additional transformation applied to it.
		for x in range(self.dimension):
			for y in range(self.dimension):
				if self._get_with_move_and_transform((x, y), move, -1) != \
				   self._get_with_move_and_transform((x, y), trans_move, -1, x_trans, y_trans):
					return False

		return True

	def _moves_equal_with_rotations(self, move1, move2, x_trans=(1, 0)):
		# To rotate a coordinate clockwise in increments of
		# 90 degrees, the following transformations need
		# to be done:
		#
		# 0 degrees:   (x, y) -> (x, y)
		# 90 degrees:  (x, y) -> (-y, x)
		# 180 degrees: (x, y) -> (-x, -y)
		# 270 degrees: (x, y) -> (y, -x)
		#
		# What this essentially is is the transformation
		# (x, y) -> (-y, x) applied a certain number of times.
		# Thus, the same effect could probably be had by
		# applying that operation the necessary number of times.
		# However, I thought it would be faster to spell them
		# all out.
		#
		# Note that, because we are ultimately using
		# _get_with_move_and_transform here, we are actually
		# rotating our VIEW of the board. This is the equivalent
		# of rotating your head by the given amount, as opposed
		# to rotating the piece of paper that the board is
		# written on. Thus, the board effectively gets rotated
		# counter-clockwise, even though we are transforming
		# the coordinates clockwise.
		x_coeff, x_add = x_trans

		# These functions are mappings that need to be applied to x
		# in order to apply rotation transformations. The x_coeff
		# and x_adds are to support an addition horizontal reflection.
		x_lambdas = [
			# identity rotation (0 degrees)
			lambda x, y: x_coeff * x + x_add,
			# first rotation (90 degrees)
			lambda x, y: x_coeff * (self.dimension - y - 1) + x_add,
			# second rotation (180 degrees)
			lambda x, y: x_coeff * (self.dimension - x - 1) + x_add,
			# third rotation (270 degrees)
			lambda x, y: x_coeff * y + x_add,
		]

		# These functions are mappings that need to be applied to y
		# in order to apply rotations transformations.
		y_lambdas = [
			# identity rotation (0 degrees)
			lambda x, y: y,
			# first rotation (90 degrees)
			lambda x, y: x,
			# second rotation (180 degrees)
			lambda x, y: self.dimension - y - 1,
			# third rotation (270 degrees)
			lambda x, y: self.dimension - x - 1
		]

		# Check whether the board with the first move made is the same
		# as the board with the second move made plus any of the four
		# rotation transformations.
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

		# Check all of the rotations with and without a horizontal reflection
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

	def get_unique_moves(self):
		"""
		Get a list of of unique moves, taking into account
		rotations and reflections. This will be a subset
		of the moves returned by get_valid_moves.
		"""
		possible_moves = self.get_valid_moves()
		unique_moves = []

		for px, py in possible_moves:
			for ex, ey in unique_moves:
				if self.moves_equal((px, py), (ex, ey)):
					break
			else:
				unique_moves.append((px, py))

		return unique_moves

TreeNode = namedtuple('TreeNode', 'scores dist moves')

class Game:
	"""
	The class that encapsulates the tic-tac-toe game logic.
	"""

	symbols = ['X', 'O', 'Y', 'Z']

	def _gen_moves_tree(self, board=None, player=0):
		if board is None:
			board = Board(self.board.dimension, 0, 0)

		moves = board.get_unique_moves()
		if len(moves) == 0:
			raise TicTacToeException('There are no moves to make from here!')

		players = len(self.players)

		best_move = None
		best_move_dist = 0

		treedict = {}
		# For each unique move...
		for x, y in moves:
			board.put(x, y, player)

			subnode = None

			# Generate a cat's game node, if necessary.
			# In a cat's game, all players will have
			# a score of 1
			if self._check_cats_game(board):
				subnode = TreeNode(dist=0,
				                   scores=[1] * players,
				                   moves=None)

			if subnode is None:
				# Generate a win game node, if necessary.
				# In a winning state, the winner has score
				# 2 and every other player has score 0.
				winner = self._check_win(x, y, board)
				if winner is not None:
					scores = [0] * players
					scores[player] = 2
					subnode = TreeNode(dist=0,
					                   scores=scores,
					                   moves=None)

			if subnode is None:
				# Otherwise, expand this node further.
				subnode = self._gen_moves_tree(board, (player + 1) % players)

			# Keep track of what the best move would be for the
			# current player, and use that as the score for the
			# node currently being expanded. This is the core idea
			# behind the minimax algorithm.
			if best_move is None or best_move[player] < subnode.scores[player] or \
			   (best_move[player] == subnode.scores[player] and
			    best_move_dist < subnode.dist):
				best_move = subnode.scores
				best_move_dist = subnode.dist

			treedict[(x, y)] = subnode

			board.undo()

		return TreeNode(dist=best_move_dist + 1,
		                scores=best_move,
		                moves=treedict)

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

		self.tree = self._gen_moves_tree()
		self.tree_board = Board(dimension, 0, 0)

		self._clear_state()

	def _clear_state(self):
		self.board.clear()
		self.cur_player = None
		self.last_move = None

		self.cur_tree = self.tree
		self.tree_board.clear()

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

			# Traverse the game tree - figure out which path we just took
			for (x, y), node in self.cur_tree.moves.items():
				self.tree_board.put(x, y, self.cur_player)

				if self.board.is_equivalent(self.tree_board):
					self.cur_tree = node
					break

				self.tree_board.undo()

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
	"""
	This class allows a human player to play tic-tac-toe by entering
	moves at a prompt.
	"""

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
	"""
	This class plays tic-tac-toe automatically by examining
	the minimax game tree generated by the Game class.
	"""

	def __init__(self, noises=['Beep', 'Boop']):
		self.noises = noises

	def get_move(self, game):
		# Generate some computer noises :)
		for noise in islice(cycle(self.noises), randint(3, 6)):
			sys.stdout.write(noise + '... ')
			sys.stdout.flush()
			time.sleep(0.1)

		print()
		print()

		node = game.cur_tree

		# The best moves we can make have this score
		best_score = max(move.scores[game.cur_player] for move in node.moves.values())

		if best_score == 2:
			print('Ha Ha Ha I can beat you now!')
			time.sleep(0.5)
			print()

		# Among the moves that get us that score, what is the best distance to endgame?
		best_dist = min(move.dist for move in node.moves.values()
		                          if move.scores[game.cur_player] == best_score)

		# Collect the moves that have the properties determined above
		best_moves = []
		for point, move_node in node.moves.items():
			if move_node.scores[game.cur_player] == best_score and \
			   move_node.dist == best_dist:
				best_moves.append(point)

		# Choose one of these moves randomly
		ex, ey = choice(best_moves)

		# Figure out the equivalent moves that the chosen move maps to
		game.tree_board.put(ex, ey, game.cur_player)

		equiv_moves = []
		for x, y in game.board.get_valid_moves():
			game.board.put(x, y, game.cur_player)

			if game.board.is_equivalent(game.tree_board):
				equiv_moves.append((x, y))

			game.board.undo()

		game.tree_board.undo()

		# Choose randomly from these equivalent moves
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
