#include <stdlib.h>
#include "$HEADER_NAME"

$BODY_PREAMBULE
void ${NAME}_init($NAME *self)
{
	self->ptr = NULL;
	self->size = self->capacity = 0;
}


void ${NAME}_dealloc($NAME *self)
{
	${NAME}_clear(self);
	$FREE(self->ptr);
}


void ${NAME}_resize($NAME *self, size_t capacity)
{
	if (self->size < capacity)
	{
		$T *ptr = ($T *)$MALLOC(sizeof(self->ptr[0]) * capacity);

		for (size_t i = 0; i < self->size; i += 1)
			ptr[i] = self->ptr[i];

		$FREE(self->ptr);

		self->ptr = ptr;
		self->capacity = capacity;
	}
	else
	{
		for (size_t i = capacity; i < self->size; i += 1)
			$DESTRUCT(self->ptr[i]);

		self->size = self->capacity = capacity;
	}
}


void ${NAME}_append($NAME *self, $T elem)
{
	if (self->size == self->capacity)
	{
		size_t size = (self->capacity < $MIN_CAPACITY) ? $MIN_CAPACITY : self->capacity * 2;
		${NAME}_resize(self, size);
	}

	self->ptr[self->size] = elem;
	self->size += 1;
}


bool ${NAME}_pop($NAME *self, $T *elem)
{
	if (self->size == 0)
		return false;

	self->size -= 1;
	*elem = self->ptr[self->size];

	return true;
}


void ${NAME}_clear($NAME *self)
{
	for (size_t i = 0; i < self->size; i += 1)
		$DESTRUCT(self->ptr[i]);

	self->size = 0;
}


bool ${NAME}_index($NAME *self, size_t index, $T *elem)
{
	if (index >= self->size)
		return false;

	*elem = self->ptr[index];
	return true;
}
