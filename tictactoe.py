#!/usr/bin/python

from itertools import *
import sys
import re
import time
from random import randint
from copy import deepcopy
from collections import namedtuple

class TicTacToeException(Exception):
	pass

class Board:
	def __init__(self, dimension, hpad, vpad):
		self.dimension = dimension

		self.hpad = hpad
		self.vpad = vpad
		self.pre_width = self._min_width(self.dimension)

		self.clear()

	def clear(self, data=None):
		self.board = [[None] * self.dimension for i in range(self.dimension)]

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

	def put(self, x, y, val):
		if x >= self.dimension or y >= self.dimension:
			raise TicTacToeException('Those values are off the board!')

		if self.board[x][y] is not None:
			raise TicTacToeException('This space is already occupied!')

		self.board[x][y] = val

	def row(self, num):
		return (self.board[i][num] for i in range(self.dimension))

	def col(self, num):
		return self.board[num]

	def _diag(self, a=1, b=0):
		return (self.board[i][a*i + b] for i in range(self.dimension))

	def main_diag(self):
		return self._diag()

	def anti_diag(self):
		return self._diag(-1, -1)

	def get_valid_moves(self):
		return ((x, y) for x in range(self.dimension)
		               for y in range(self.dimension)
		               if self.board[x][y] is None)

	def rotate(self):
		self.board = [list(self.row(self.dimension - i - 1)) for i in range(self.dimension)]

	def reflect_horiz(self):
		self.board.reverse()

	def reflect_vert(self):
		self.board = [col.reverse() for col in self.board]

	def __eq__(self, other):
		return (self.dimension == other.dimension and
		        self.board == other.board)

	def is_equivalent(self, other):
		for i in range(4):
			if self == other:
				return True

			other.rotate()

		other.reflect_horiz()

		for i in range(4):
			if self == other:
				return True

			other.rotate()

		other.reflect_horiz()

		return False

	def get_equivalent_moves(self):
		possible_moves = self.get_valid_moves()
		equivalent_moves = []

		for px, py in possible_moves:
			pboard = deepcopy(self)
			pboard.put(px, py, -1)

			already_there = False
			for ex, ey in equivalent_moves:
				eboard = deepcopy(self)
				eboard.put(ex, ey, -1)

				if pboard.is_equivalent(eboard):
					already_there = True
					break

			if not already_there:
				equivalent_moves.append((px, py))

		return equivalent_moves

GraphNode = namedtuple('GraphNode', 'scores dist board moves')

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
			dup = deepcopy(board)
			dup.put(x, y, player)

			subnode = None
			if self._check_cats_game(dup):
				scores = [1] * players
				subnode = GraphNode(dist=0,
				                    scores=scores,
				                    board=dup,
				                    moves={})

			if subnode is None:
				winner = self._check_win(x, y, dup)
				if winner is not None:
					scores = [0] * players
					scores[player] = 2
					subnode = GraphNode(dist=0,
					                    scores=scores,
					                    board=dup,
					                    moves={})

			if subnode is None:
				subnode = self._gen_moves_graph(dup, (player + 1) % players)

			if best_move is None or best_move[player] < subnode.scores[player]:
				best_move = subnode.scores
				best_move_dist = subnode.dist + 1

			graphdict[(x, y)] = subnode

		return GraphNode(dist=best_move_dist,
		                 scores=best_move,
		                 board=board,
		                 moves=graphdict)

	def __init__(self, *players, dimension=3, hpad=2, vpad=1):
		if len(players) > len(self.symbols):
			raise TicTacToeException('Too many players!')

		self.players = players
		self.board = Board(dimension, hpad, vpad)

		self.graph = self._gen_moves_graph()

		self._clear_state()

	def _clear_state(self):
		self.board.clear()
		self.cur_player = None
		self.last_move = None
		self.cur_graph = self.graph

	def run(self):
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

			for point, node in self.cur_graph.moves.items():
				if self.board.is_equivalent(node.board):
					self.cur_graph = node
					break

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

		x, y = next(game.board.get_valid_moves())

		# Let the exceptions go unhandled - nothing we can really do about them
		game.do_move(x, y)

def main():
	player1 = ManualPlayer()
	player2 = AutomaticPlayer()

	game = Game(player1, player2)

	while True:
		game.run()

		prompt = input('Play again (say "yes" for affirmative)? ').lower()
		if not 'yes'.startswith(prompt):
			print('Fine. Bye.')
			break;

		print()

if __name__ == '__main__':
	main()
