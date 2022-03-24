happy(X) :-
  rich(X).
happy(X) :-
  man(X),woman(Y),likes(X,Y),likes(Y,X).
happy(Y) :-
  man(X),woman(Y),likes(X,Y),likes(Y,X).
likes(X,Y) :-
  man(X),woman(Y),pretty(Y).
likes(catherine,X) :-
  man(X),likes(X,catherine).
likes(helen,X) :-
  man(X),kind(X),rich(X).
likes(helen,X) :-
  man(X),muscly(X),pretty(X).
muscly(tim).
muscly(peter).
kind(tim).
rich(peter).
rich(john).
pretty(john).
pretty(helen).
man(tim).
man(peter).
man(john).
woman(helen).
woman(catherine).
