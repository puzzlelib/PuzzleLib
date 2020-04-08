#undef NDEBUG
#include <assert.h>

#include "AllocTree.gen.h"
#include "$HEADER_NAME"


static AllocTree allocTree;
static AllocTree_Iterator allocIterator;


void *${NAME}_malloc(size_t size, const char *file, int line)
{
	void *ptr = malloc(size);

	Allocation alloc;
	alloc.size = size;
	alloc.file = file;
	alloc.line = line;

	bool inserted = AllocTree_insert(&allocTree, ptr, alloc);
	assert(inserted);

	return ptr;
}


void ${NAME}_free(void *ptr)
{
	if (ptr != NULL)
	{
		Allocation alloc;

		bool found = AllocTree_get(&allocTree, ptr, &alloc);
		assert(found);

		bool deleted = AllocTree_delete(&allocTree, ptr);
		assert(deleted);
	}

	free(ptr);
}


size_t ${NAME}_traceLeaks(void)
{
	return allocTree.size;
}


bool ${NAME}_Iterator_init(void)
{
	return AllocTree_Iterator_init(&allocIterator, &allocTree, true);
}


void ${NAME}_Iterator_dealloc(void)
{
	AllocTree_Iterator_dealloc(&allocIterator);
}


bool ${NAME}_Iterator_move(void)
{
	return AllocTree_Iterator_move(&allocIterator, true);
}


void ${NAME}_Iterator_item(size_t *size, const char **file, int *line)
{
	void *ptr;
	Allocation alloc;

	AllocTree_Iterator_item(&allocIterator, &ptr, &alloc);

	*size = alloc.size;
	*file = alloc.file;
	*line = alloc.line;
}
