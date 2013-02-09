#!/usr/bin/python

from itertools import *
import sys
import re
import time
from random import randint

class TicTacToeGame:
	symbols = ['X', 'O', 'Y', 'Z']

	def __init__(self, *players, dimension=3, hpad=2, vpad=1):
		if len(players) > len(self.symbols):
			raise Exception('Too many players!')

		self.players = players
		self.dimension = dimension

		self.hpad = hpad
		self.vpad = vpad
		self.pre_width = self._min_width(self.dimension)

		self._clear_state()

	def _clear_state(self):
		self.board = [[None] * self.dimension for i in range(self.dimension)]
		self.cur_player = None
		self.last_move = None

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

	def _print_board(self):
		def objs_to_symbols(players):
			return (self.symbols[player] for player in players if player is not None)

		def places_to_symbols(places):
			for place in places:
				if place is not None:
					yield self.symbols[place]
				else:
					yield place

		column_widths = [max(self._min_width(val) for val in chain(objs_to_symbols(column), [i]))
		                 for i, column in enumerate(self.board, 1)]

		self._print_board_row(None, ' ', range(1, self.dimension+1), column_widths)

		for i in range(self.dimension):
			self._print_board_line(None, '+', ('-' * width for width in column_widths))
			self._print_board_row(str(i+1), '|', places_to_symbols(column[i] for column in self.board), column_widths)

		self._print_board_line(None, '+', ('-' * width for width in column_widths))

	def run(self):
		print('The initial state of the board is:')
		self._print_board()
		print()

		for i in cycle(range(len(self.players))):
			self.cur_player = i
			print('It is {0}\'s turn!'.format(self.symbols[i]))
			print()

			self.players[i].get_move(self)

			if self.last_move is None:
				raise Exception('No move was made???')

			x, y = self.last_move
			self.last_move = None

			print()
			print('The state of the board is:')
			self._print_board()
			print()

			winner = self._check_win(x, y)

			if winner is not None:
				print('We have a winner! It is {0}! Congratulations!'.format(self.symbols[winner]))
				break

			if self._check_cats_game():
				print('Looks like nobody can win now - it\'s a tie! \'round these parts, we call that a "cat\'s game".')
				break

		self._clear_state()

	def _check_cats_game_sequence(self, seq):
		grouped = list(groupby(val for val in seq
		                           if val is not None))

		return len(grouped) > 1

	def _check_cats_game(self):
		all_seqs = chain(
			(self._row(i) for i in range(self.dimension)),
			(self._col(i) for i in range(self.dimension)),
			[self._tl_br_diag(), self._bl_tr_diag()]
		)

		return all(self._check_cats_game_sequence(seq) for seq in all_seqs)

	def _check_win_sequence(self, seq):
		grouped = [k for k, g in groupby(seq)]

		if len(grouped) > 1 or grouped[0] is None:
			return None
		else:
			return grouped[0]

	def _row(self, num):
		return (self.board[i][num] for i in range(self.dimension))

	def _col(self, num):
		return self.board[num]

	def _diag(self, a=1, b=0):
		return (self.board[i][a*i + b] for i in range(self.dimension))

	def _tl_br_diag(self):
		return self._diag()

	def _bl_tr_diag(self):
		return self._diag(-1, -1)

	def _check_win(self, x, y):
		to_check = [self._col(x), self._row(y)]

		# main diagonal
		if x == y:
			to_check += [self._tl_br_diag()]

		# anti-diagonal
		if x == (self.dimension - 1) - y:
			to_check += [self._bl_tr_diag()]

		for sequence in to_check:
			player = self._check_win_sequence(sequence)

			if player is not None:
				return player

		return None

	def get_valid_moves(self):
		return ((x, y) for x in range(self.dimension)
		               for y in range(self.dimension)
		               if self.board[x][y] is None)

	def do_move(self, x, y, board=None):
		if x >= self.dimension or y >= self.dimension:
			raise Exception('Those values are off the board!')

		if self.last_move is not None:
			raise Exception('Come on man, you obviously can\'t go twice...')

		if board is None:
			board = self.board

		if board[x][y] is not None:
			raise Exception('This space is already occupied!')

		board[x][y] = self.cur_player

		if board is self.board:
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
			except Exception as e:
				print(e.args[0])
				continue
			else:
				break

class AutomaticPlayer:
	def __init__(self, noises=['Beep', 'Boop']):
		self.noises = noises

	def get_move(self, game):
		for noise in islice(cycle(self.noises), randint(5, 15)):
			sys.stdout.write(noise + '... ')
			sys.stdout.flush()
			time.sleep(0.2)

		print()
		print()

		x, y = next(game.get_valid_moves())

		# Let the exceptions go unhandled - nothing we can really do about them
		game.do_move(x, y)

player1 = ManualPlayer()
player2 = AutomaticPlayer()

game = TicTacToeGame(player1, player2)

while True:
	game.run()

	prompt = input('Play again (say "yes" for affirmative)? ').lower()
	if not 'yes'.startswith(prompt):
		print('Fine. Bye.')
		break;

	print()
