package main

type linkedListNode[T any] struct {
	value T
	next  *linkedListNode[T]
}

type linkedList[T any] struct {
	head *linkedListNode[T]
	tail *linkedListNode[T]
	size int
}

type Deque[T any] interface {
	Push(e T)
	Pop() T
	Size() int
}

type Stack[T any] struct {
	list *linkedList[T]
}

type Queue[T any] struct {
	list *linkedList[T]
}

func NewStack[T any]() *Stack[T] {
	return &Stack[T]{
		list: &linkedList[T]{},
	}
}

func (s *Stack[T]) Push(e T) {
	newNode := &linkedListNode[T]{value: e}
	if s.list.size == 0 {
		s.list.head = newNode
		s.list.tail = newNode
	} else {
		newNode.next = s.list.head
		s.list.head = newNode
	}
	s.list.size++
}

func (s *Stack[T]) Pop() T {
	if s.list.size == 0 {
		var zero T
		return zero
	}
	node := s.list.head
	s.list.head = s.list.head.next
	s.list.size--
	if s.list.size == 0 {
		s.list.tail = nil
	}
	return node.value
}

func (s *Stack[T]) Size() int {
	return s.list.size
}

func NewQueue[T any]() *Queue[T] {
	return &Queue[T]{
		list: &linkedList[T]{},
	}
}

func (q *Queue[T]) Push(e T) {
	newNode := &linkedListNode[T]{value: e}
	if q.list.size == 0 {
		q.list.head = newNode
		q.list.tail = newNode
	} else {
		q.list.tail.next = newNode
		q.list.tail = newNode
	}
	q.list.size++
}

func (q *Queue[T]) Pop() T {
	if q.list.size == 0 {
		var zero T
		return zero
	}
	node := q.list.head
	q.list.head = q.list.head.next
	q.list.size--
	if q.list.size == 0 {
		q.list.tail = nil
	}
	return node.value
}

func (q *Queue[T]) Size() int {
	return q.list.size
}
