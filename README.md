# ScramblingProject
Проект реализует механизм регуляции авторских прав для изображений, а конкретно скремблирует или дескремблирует их по запросу пользователя.
  
Скремблирование изображения происходит таким образом, что:
1. Изображение может быть декодировано с помощью стандартного декодера, но в плохом качестве.
2. Вы можете получить изображение в максимальном качестве только если знаете секретный ключ.
   
Функции скремблирования и дескремблирования принимают на вход:
- само изображение (формат – .png);
- зерно генератора случайных чисел;
- вероятность p;
- число n.

Из этих параметров ключом являются зерно, p и n. Таким образом, без ключа
пользователь будет видеть изображение в плохом качестве, а после покупки
ключа сможет декодировать изображение и видеть его в отличном качестве.

Прямо сейчас для корректной работы кода картинка должна находиться в одном каталоге с кодом, называться "Protorave" и иметь расширение ".png".
