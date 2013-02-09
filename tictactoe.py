#!/usr/bin/python

from itertools import chain, cycle, groupby
import sys
import re

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

	def _print_board_line(self, before, sep, things):
		if before is None:
			before = ' ' * self.pre_width
		else:
			before = before.center(self.pre_width)

		print(before + sep + sep.join(things) + sep)

	def _things(self, things, widths):
		for (i, val) in enumerate(things):
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
		                 for (i, column) in enumerate(self.board, 1)]

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
			print('It is {0}\'s turn!'.format(self.symbols[i]))
			print()

			player = self.players[i]

			while True:
				x, y = player.get_move(self)

				if x >= self.dimension or y >= self.dimension:
					player.off_the_board()
					continue

				if self.board[x][y] is not None:
					player.already_occupied()
					continue

				break

			self.board[x][y] = i

			print('The state of the board is:')
			self._print_board()
			print()

			winner = self._check_win(x, y)

			if winner is not None:
				print('We have a winner! It is you, {0}!'.format(self.symbols[winner]))
				break

			if self._check_cats_game():
				print('Looks like nobody can win now - it\'s a tie! \'round these parts, we call that a "cat\'s game".')
				break

		self._clear_state()

	def _check_cats_game_sequence(self, seq):
		grouped = [k for k, g in groupby(val for val in seq if val is not None)]

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



class ManualPlayer:
	def __init__(self):
		self.input_re = re.compile('^[^\d]*(\d+)[^\d]+(\d+)[^\d]*$')

	def off_the_board(self):
		print('Those values are off the board!')

	def already_occupied(self):
		print('This space is already occupied!')

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

			break

		return x - 1, y - 1

player1 = ManualPlayer()
player2 = ManualPlayer()
player3 = ManualPlayer()
player4 = ManualPlayer()

game = TicTacToeGame(player1, player2, player3, player4)

while True:
	game.run()

	prompt = input('Play again? ').lower()
	if not 'yes'.startswith(prompt):
		print('Fine. Bye.')
		break;
