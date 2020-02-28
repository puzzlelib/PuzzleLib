class Type:
	precedence = None
	cache = None


	@property
	def ptr(self):
		return getCached(PointerType, self)


	def __getitem__(self, item):
		return getCached(ArrayType, self, item)


	def func(self, *argtypes):
		return getCached(FunctionType, self, argtypes)


	@property
	def unqual(self):
		T = self

		while isinstance(T, QualifierType):
			T = T.basetype

		return T


	@property
	def const(self):
		return getCached(ConstType, self)


	@property
	def restrict(self):
		return getCached(RestrictType, self)


	@property
	def volatile(self):
		return getCached(VolatileType, self)


	def typegen(self, asDecl):
		raise NotImplementedError()


	@property
	def initializer(self):
		raise NotImplementedError()


	@property
	def aliasBase(self):
		return self


	def basedWith(self, T):
		return T


	def __str__(self):
		return self.typegen(asDecl=False)


	def __repr__(self):
		return self.typegen(asDecl=False)


class QualifierType(Type):
	precedence = 0


	def __init__(self, basetype):
		self.basetype = basetype


	def typegen(self, asDecl):
		T = self.qualifier + " %s" if asDecl else self.qualifier
		return self.basetype.typegen(asDecl=True) % T


	@property
	def initializer(self):
		return self.basetype.initializer


	@property
	def aliasBase(self):
		return self.basetype.aliasBase


	@property
	def qualifier(self):
		raise NotImplementedError()


class ConstType(QualifierType):
	cache = {}


	@property
	def qualifier(self):
		return "const"


	def basedWith(self, T):
		return self.basetype.basedWith(T).const


class VolatileType(QualifierType):
	cache = {}


	@property
	def qualifier(self):
		return "volatile"


	def basedWith(self, T):
		return self.basetype.basedWith(T).volatile


class RestrictType(QualifierType):
	cache = {}


	@property
	def qualifier(self):
		return "__restrict"


	def basedWith(self, T):
		return self.basetype.basedWith(T).restrict


class PointerType(Type):
	precedence = 1
	cache = {}


	def __init__(self, basetype):
		self.basetype = basetype


	def typegen(self, asDecl):
		T = "*%s" if asDecl else "*"

		T = "(%s)" % T if self.precedence < self.basetype.precedence else T
		return self.basetype.typegen(asDecl=True) % T


	@property
	def initializer(self):
		return "NULL"


	@property
	def aliasBase(self):
		return self.basetype.aliasBase.ptr


	def basedWith(self, T):
		return self.basetype.basedWith(T).ptr


class ArrayType(Type):
	precedence = 2
	cache = {}


	def __init__(self, elemtype, size):
		self.elemtype = elemtype
		self.size = size


	def typegen(self, asDecl):
		T = ("%s" if asDecl else "") + "[%s]" % ("" if self.size is None else self.size)

		T = "(%s)" % T if self.precedence < self.elemtype.precedence else T
		return self.elemtype.typegen(asDecl=True) % T


	@property
	def initializer(self):
		return "{%s}" % self.elemtype.initializer


	@property
	def aliasBase(self):
		return self.elemtype.aliasBase[self.size]


	def basedWith(self, T):
		return self.elemtype.basedWith(T)[self.size]


class FunctionType(Type):
	precedence = 1
	cache = {}


	def __init__(self, returntype, argtypes):
		self.returntype = returntype
		self.argtypes = argtypes


	def typegen(self, asDecl):
		argtypes = "(%s)" % ", ".join(argtype.typegen(asDecl=False) for argtype in self.argtypes)
		T = ("(*%s)" if asDecl else "(*)") + argtypes

		T = "(%s)" % T if self.precedence < self.returntype.precedence else T
		return self.returntype.typegen(asDecl=True) % T


	@property
	def initializer(self):
		return "NULL"


	@property
	def aliasBase(self):
		return self.returntype.aliasBase(argtype.aliasBase for argtype in self.argtypes)


	def basedWith(self, T):
		return self.returntype.basedWith(T).func(*self.argtypes)


class VoidType(Type):
	precedence = 0


	def typegen(self, asDecl):
		return "void" + (" %s" if asDecl else "")


	@property
	def initializer(self):
		return None


class NumberType(Type):
	precedence = 0


	def __init__(self, name, initvalue):
		self.name = name
		self.initvalue = initvalue


	def typegen(self, asDecl):
		return self.name + (" %s" if asDecl else "")


	@property
	def initializer(self):
		return self.initvalue


def getCached(cls, *args):
	T = cls.cache.get(args, None)

	if T is None:
		T = cls(*args)
		cls.cache[args] = T

	return T


void_t = VoidType()
bool_t = NumberType("bool", initvalue="false")

char_t = NumberType("char", initvalue="'\\0'")
wchar_t = NumberType("wchar_t", initvalue="L'\\0'")

schar_t = NumberType("signed char", initvalue="0")
uchar_t = NumberType("unsigned char", initvalue="0U")

short_t = NumberType("short", initvalue="0")
short2_t = NumberType("short2", initvalue="short2()")
ushort_t = NumberType("unsigned short", initvalue="0U")
ushort2_t = NumberType("ushort2", initvalue="ushort2()")

int_t = NumberType("int", initvalue="0")
uint_t = NumberType("unsigned int", initvalue="0U")

long_t = NumberType("long", initvalue="0L")
ulong_t = NumberType("unsigned long", initvalue="0UL")

llong_t = NumberType("long long", initvalue="0LL")
ullong_t = NumberType("unsigned long long", initvalue="0ULL")

half_t = NumberType("half", initvalue="0")
half2_t = NumberType("half2", initvalue="half2()")

float_t = NumberType("float", initvalue="0.0f")
double_t = NumberType("double", initvalue="0.0")
ldouble_t = NumberType("long double", initvalue="0.0L")


int8_t = NumberType("int8_t", initvalue="0")
uint8_t = NumberType("uint8_t", initvalue="0U")

int16_t = NumberType("int16_t", initvalue="0")
uint16_t = NumberType("uint16_t", initvalue="0U")

int32_t = NumberType("int32_t", initvalue="0")
uint32_t = NumberType("uint32_t", initvalue="0U")

int64_t = NumberType("int64_t", initvalue="0LL")
uint64_t = NumberType("uint64_t", initvalue="0ULL")

ptrdiff_t = NumberType("ptrdiff_t", initvalue="0")
size_t = NumberType("size_t", initvalue="0")

Py_ssize_t = NumberType("Py_ssize_t", initvalue="0")
