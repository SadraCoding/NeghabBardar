import 'dart:convert';
import 'dart:io';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter_acrylic/flutter_acrylic.dart' as acrylic;
import 'package:bitsdojo_window/bitsdojo_window.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'package:url_launcher/url_launcher.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await acrylic.Window.initialize();
  await acrylic.Window.setEffect(
    effect: acrylic.WindowEffect.acrylic,
    color: const Color(0xAA101010),
  );
  runApp(const MyApp());
  doWhenWindowReady(() {
    final win = appWindow;
    win.minSize = const Size(960, 620);
    win.maxSize = const Size(960, 620);
    win.size = const Size(960, 620);
    win.alignment = Alignment.center;
    win.title = "NeghabBardar";
    win.show();
  });
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return Directionality(
      textDirection: TextDirection.rtl,
      child: MaterialApp(
        debugShowCheckedModeBanner: false,
        title: 'NeghabBardar',
        theme: ThemeData.dark(useMaterial3: true).copyWith(
          textTheme: ThemeData.dark().textTheme.apply(fontFamily: 'Vazir'),
        ),
        home: const DeepFakeScreen(),
      ),
    );
  }
}

class DeepFakeScreen extends StatefulWidget {
  const DeepFakeScreen({super.key});
  @override
  State<DeepFakeScreen> createState() => _DeepFakeScreenState();
}

class _DeepFakeScreenState extends State<DeepFakeScreen>
    with SingleTickerProviderStateMixin {
  String? resultText;
  bool isLoading = false;
  late AnimationController _controller;
  late Animation<double> _scaleAnim;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
        duration: const Duration(milliseconds: 700), vsync: this);
    _scaleAnim = CurvedAnimation(parent: _controller, curve: Curves.easeOutBack);
  }

  Future<void> _pickImageAndSend() async {
    final picked = await FilePicker.platform.pickFiles(type: FileType.image);
    if (picked != null && picked.files.single.path != null) {
      final file = File(picked.files.single.path!);
      setState(() {
        isLoading = true;
        resultText = null;
      });
      final request = http.MultipartRequest(
          'POST', Uri.parse("http://127.0.0.1:8000/predict"))
        ..files.add(await http.MultipartFile.fromPath('file', file.path));
      try {
        final response = await request.send();
        final body = await response.stream.bytesToString();
        if (response.statusCode == 200) {
          final decoded = json.decode(body);
          setState(() {
            resultText = decoded['prediction'];
          });
          _controller.forward(from: 0);
        } else {
          setState(() {
            resultText = "خطا در پاسخ سرور";
          });
        }
      } catch (_) {
        setState(() {
          resultText = "اتصال برقرار نشد";
        });
      } finally {
        setState(() {
          isLoading = false;
        });
      }
    }
  }

  Color _getResultColor(String? text) {
    if (text == null) return Colors.white;
    if (text.contains('واقعی')) return Colors.greenAccent;
    if (text.contains('جعلی')) return Colors.redAccent;
    return Colors.amberAccent;
  }

  IconData _getResultIcon(String? text) {
    if (text == null) return Icons.help_outline_rounded;
    if (text.contains('واقعی')) return Icons.verified_rounded;
    if (text.contains('جعلی')) return Icons.warning_amber_rounded;
    return Icons.info_outline_rounded;
  }

  Future<void> _launchURL() async {
    final Uri url = Uri.parse('https://sadramilani.ir/fa');
    if (!await launchUrl(url, mode: LaunchMode.externalApplication)) {
      throw 'Could not launch $url';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Stack(children: [
        Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [Color(0xFF141519), Color(0xFF1A1A22)],
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
            ),
          ),
        ),
        Center(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(35),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 35, sigmaY: 35),
              child: Container(
                width: 720,
                padding: const EdgeInsets.all(36),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.35),
                  borderRadius: BorderRadius.circular(35),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.35),
                      blurRadius: 34,
                      spreadRadius: 1,
                      offset: const Offset(0, 8),
                    ),
                  ],
                  border: Border.all(
                    color: Colors.blueAccent.withOpacity(0.16),
                    width: 1.8,
                  ),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(
                      Icons.face_retouching_natural_sharp,
                      color: Colors.cyanAccent,
                      size: 54,
                    ),
                    const SizedBox(height: 8),
                    ShaderMask(
                      shaderCallback: (rect) => const LinearGradient(
                        colors: [Colors.cyanAccent, Colors.lightBlueAccent],
                      ).createShader(rect),
                      child: const Text(
                        "نقاب‌بردار | تشخیص چهره های غیر واقعی",
                        style: TextStyle(
                          fontSize: 32,
                          fontWeight: FontWeight.w900,
                          color: Colors.white,
                          letterSpacing: 1,
                          shadows: [
                            Shadow(
                                color: Colors.black54,
                                blurRadius: 7,
                                offset: Offset(0, 2))
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 18),
                    const Text(
                      "تصویر یک چهره را بارگذاری کنید تا سیستم بررسی کند که واقعی است یا جعلی",
                      style: TextStyle(
                          fontSize: 19,
                          color: Colors.white70,
                          fontWeight: FontWeight.bold),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 32),
                    MouseRegion(
                      cursor: SystemMouseCursors.click,
                      child: GestureDetector(
                        onTap: isLoading ? null : _pickImageAndSend,
                        child: AnimatedContainer(
                          duration: const Duration(milliseconds: 180),
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(22),
                            gradient: isLoading
                                ? LinearGradient(colors: [
                                    Colors.grey.shade500,
                                    Colors.grey.shade700
                                  ])
                                : const LinearGradient(
                                    colors: [
                                        Color(0xFF20CDFA),
                                        Color(0xFF013DFD)
                                      ],
                                    begin: Alignment.centerRight,
                                    end: Alignment.centerLeft,
                                  ),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.blueAccent.withOpacity(.23),
                                blurRadius: 10,
                                offset: const Offset(0, 3),
                              )
                            ],
                          ),
                          padding: const EdgeInsets.symmetric(
                              horizontal: 42, vertical: 18),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(Icons.image_search_outlined,
                                  size: 25, color: Colors.white),
                              const SizedBox(width: 13),
                              const Text(
                                "انتخاب تصویر",
                                style: TextStyle(
                                    fontSize: 19,
                                    fontWeight: FontWeight.w600,
                                    color: Colors.white),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 34),
                    AnimatedSwitcher(
                      duration: const Duration(milliseconds: 320),
                      child: isLoading
                          ? const CircularProgressIndicator()
                          : resultText != null
                              ? ScaleTransition(
                                  scale: _scaleAnim,
                                  child: Container(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 20, vertical: 21),
                                    decoration: BoxDecoration(
                                      color: Colors.white.withOpacity(0.10),
                                      borderRadius: BorderRadius.circular(18),
                                      border: Border.all(
                                          color: Colors.white30, width: 1.2),
                                    ),
                                    child: Row(
                                      mainAxisSize: MainAxisSize.min,
                                      mainAxisAlignment:
                                          MainAxisAlignment.center,
                                      children: [
                                        Icon(
                                          _getResultIcon(resultText),
                                          color: _getResultColor(resultText),
                                          size: 36,
                                        ),
                                        const SizedBox(width: 17),
                                        Text(
                                          "نتیجه: $resultText",
                                          style: TextStyle(
                                              fontSize: 23,
                                              fontWeight: FontWeight.bold,
                                              color:
                                                  _getResultColor(resultText),
                                              letterSpacing: 1),
                                          textAlign: TextAlign.center,
                                        ),
                                      ],
                                    ),
                                  ),
                                )
                              : const SizedBox.shrink(),
                    ),
                    const SizedBox(height: 40),
                    MouseRegion(
                      cursor: SystemMouseCursors.click,
                      child: GestureDetector(
                        onTap: _launchURL,
                        child: RichText(
                          text: const TextSpan(
                            text: 'ساخته شده توسط ',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                              fontFamily: 'Vazir',
                            ),
                            children: [
                              TextSpan(
                                text: 'صدرا میلانی مقدم',
                                style: TextStyle(
                                  color: Colors.lightBlueAccent,
                                  fontSize: 16,
                                  fontWeight: FontWeight.w600,
                                  fontFamily: 'Vazir',
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ]),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
}
