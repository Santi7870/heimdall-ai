import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:file_picker/file_picker.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(const HeimdallApp());
}

class HeimdallApp extends StatelessWidget {
  const HeimdallApp({super.key});

  final String backendUrl = "https://valiant-ashely-provisionally.ngrok-free.dev";

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Heimdall AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF121212),
        primaryColor: Colors.deepPurpleAccent,
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF1E1E1E),
          elevation: 0,
        ),
        colorScheme: const ColorScheme.dark(
          primary: Colors.deepPurpleAccent,
          secondary: Colors.tealAccent,
        ),
      ),
      home: ChatScreen(backendUrl: backendUrl),
    );
  }
}

class ChatScreen extends StatefulWidget {
  final String backendUrl;
  const ChatScreen({super.key, required this.backendUrl});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _controller = TextEditingController();
  final TextEditingController _personaController = TextEditingController(text: "Eres un asistente útil y amable.");

  final List<Map<String, dynamic>> _messages = [];
  final ImagePicker _picker = ImagePicker();
  final AudioPlayer _audioPlayer = AudioPlayer();

  bool _isLoading = false;
  bool _pdfUploaded = false;
  File? _selectedImage;

  // --- FUNCIÓN NUEVA: REINICIAR TODO (RESET) ---
  Future<void> _resetApp() async {
    setState(() => _isLoading = true);
    try {
      // 1. Decirle al servidor que borre la memoria del PDF
      await http.post(Uri.parse('${widget.backendUrl}/reset'));

      // 2. Limpiar todo en la App
      setState(() {
        _messages.clear();
        _personaController.text = "Eres un asistente útil y amable."; // Vuelve a default
        _pdfUploaded = false;
        _selectedImage = null;
        _controller.clear();
      });

      _showSnack("🧹 Sistema reiniciado: Memoria limpia.", Colors.cyanAccent);
    } catch (e) {
      _showSnack("Error al reiniciar: $e", Colors.redAccent);
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // --- SUBIR PDF ---
  Future<void> _pickAndUploadPDF() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf'],
    );

    if (result != null) {
      File file = File(result.files.single.path!);
      setState(() => _isLoading = true);

      try {
        var request = http.MultipartRequest('POST', Uri.parse('${widget.backendUrl}/upload_pdf'));
        request.files.add(await http.MultipartFile.fromPath('file', file.path));

        var res = await request.send();

        if (res.statusCode == 200) {
          _showSnack("📄 PDF Memorizado. ¡Pregunta!", Colors.green);
          setState(() => _pdfUploaded = true);
        } else {
          _showSnack("Error subida (Code: ${res.statusCode})", Colors.red);
        }
      } catch (e) {
        _showSnack("Error conexión: $e", Colors.red);
      } finally {
        setState(() => _isLoading = false);
      }
    }
  }

  // --- ENVIAR MENSAJE ---
  Future<void> _sendMessage() async {
    if (_controller.text.isEmpty && _selectedImage == null) return;

    final String prompt = _controller.text;
    final File? img = _selectedImage;

    setState(() {
      _messages.add({"role": "user", "text": prompt, "image": img});
      _isLoading = true;
      _controller.clear();
      _selectedImage = null;
    });

    try {
      var request = http.MultipartRequest('POST', Uri.parse('${widget.backendUrl}/chat'));
      request.fields['prompt'] = prompt.isEmpty ? "Describe esto" : prompt;
      request.fields['personality'] = _personaController.text;

      if (img != null) {
        request.files.add(await http.MultipartFile.fromPath('image', img.path));
      }

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = jsonDecode(utf8.decode(response.bodyBytes));
        setState(() {
          _messages.add({"role": "assistant", "text": data['response']});
        });
        if (data['audio'] != null) _playAudio(data['audio']);
      } else {
        _showSnack("Error servidor: ${response.statusCode}", Colors.orange);
      }
    } catch (e) {
      _showSnack("Error: $e", Colors.red);
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _playAudio(String b64) async {
    try {
      final bytes = base64.decode(b64);
      final dir = await getTemporaryDirectory();
      final file = File('${dir.path}/audio_temp.mp3');
      await file.writeAsBytes(bytes);
      await _audioPlayer.play(DeviceFileSource(file.path));
    } catch (e) { print("Error audio: $e"); }
  }

  void _showSnack(String msg, Color color) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg), backgroundColor: color, behavior: SnackBarBehavior.floating));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      drawer: Drawer(
        backgroundColor: const Color(0xFF1E1E1E),
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: 60),
              const Icon(Icons.psychology, size: 60, color: Colors.deepPurpleAccent),
              const SizedBox(height: 10),
              const Center(child: Text("Configuración Mental", style: TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold))),
              const SizedBox(height: 30),
              const Text("Define la personalidad:", style: TextStyle(color: Colors.grey)),
              const SizedBox(height: 10),
              TextField(
                controller: _personaController,
                maxLines: 4,
                style: const TextStyle(color: Colors.white),
                decoration: InputDecoration(filled: true, fillColor: Colors.grey[900], border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)), hintText: "Ej: Eres un pirata espacial...", hintStyle: TextStyle(color: Colors.grey[600])),
              ),
              const SizedBox(height: 20),
              ElevatedButton.icon(
                icon: const Icon(Icons.save_as, color: Colors.white),
                label: const Text("APLICAR", style: TextStyle(color: Colors.white)),
                style: ElevatedButton.styleFrom(backgroundColor: Colors.deepPurple, padding: const EdgeInsets.symmetric(vertical: 15)),
                onPressed: () { Navigator.pop(context); _showSnack("✅ Personalidad guardada", Colors.greenAccent); },
              ),
              const Spacer(),
              if (_pdfUploaded)
                Container(
                  padding: const EdgeInsets.all(10),
                  margin: const EdgeInsets.only(bottom: 20),
                  decoration: BoxDecoration(color: Colors.green.withOpacity(0.2), borderRadius: BorderRadius.circular(10), border: Border.all(color: Colors.green)),
                  child: const Row(mainAxisAlignment: MainAxisAlignment.center, children: [Icon(Icons.check_circle, color: Colors.green), SizedBox(width: 10), Text("Memoria PDF Activa", style: TextStyle(color: Colors.green))]),
                ),
            ],
          ),
        ),
      ),
      appBar: AppBar(
        title: const Text("Heimdall AI"),
        centerTitle: true,
        actions: [
          // BOTÓN PDF
          IconButton(
            icon: Icon(Icons.picture_as_pdf, color: _pdfUploaded ? Colors.greenAccent : Colors.white),
            onPressed: _pickAndUploadPDF,
            tooltip: "Subir PDF",
          ),
          // --- BOTÓN NUEVO: RESET (PAPELERA) ---
          IconButton(
            icon: const Icon(Icons.delete_forever, color: Colors.redAccent),
            onPressed: _resetApp,
            tooltip: "Reiniciar Todo",
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.all(15),
              itemCount: _messages.length,
              itemBuilder: (context, idx) {
                final msg = _messages[idx];
                final isUser = msg['role'] == "user";
                return Align(
                  alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
                  child: Container(
                    margin: const EdgeInsets.symmetric(vertical: 5),
                    padding: const EdgeInsets.all(12),
                    constraints: const BoxConstraints(maxWidth: 300),
                    decoration: BoxDecoration(color: isUser ? Colors.deepPurple.shade700 : Colors.grey[800], borderRadius: BorderRadius.circular(15)),
                    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      if (msg['image'] != null) Padding(padding: const EdgeInsets.only(bottom: 8.0), child: Image.file(msg['image'], height: 150)),
                      Text(msg['text'], style: const TextStyle(color: Colors.white, fontSize: 16)),
                    ]),
                  ),
                );
              },
            ),
          ),
          if (_isLoading) const Padding(padding: EdgeInsets.symmetric(horizontal: 20), child: LinearProgressIndicator(color: Colors.tealAccent, backgroundColor: Colors.transparent)),
          if (_selectedImage != null) Container(margin: const EdgeInsets.symmetric(horizontal: 10), padding: const EdgeInsets.all(10), decoration: BoxDecoration(color: Colors.grey[900], borderRadius: BorderRadius.circular(10)), child: Row(children: [const Icon(Icons.image, color: Colors.tealAccent), const SizedBox(width: 10), const Expanded(child: Text("Imagen lista...", style: TextStyle(color: Colors.white70))), IconButton(icon: const Icon(Icons.close, color: Colors.redAccent), onPressed: () => setState(() => _selectedImage = null))])),
          Padding(
            padding: const EdgeInsets.all(10.0),
            child: Row(
              children: [
                Container(decoration: BoxDecoration(color: Colors.grey[800], shape: BoxShape.circle), child: IconButton(icon: const Icon(Icons.camera_alt, color: Colors.tealAccent), onPressed: () async { final img = await _picker.pickImage(source: ImageSource.gallery); if (img != null) setState(() => _selectedImage = File(img.path)); })),
                const SizedBox(width: 10),
                Expanded(child: TextField(controller: _controller, style: const TextStyle(color: Colors.white), decoration: InputDecoration(hintText: "Escribe...", filled: true, fillColor: Colors.grey[900], border: OutlineInputBorder(borderRadius: BorderRadius.circular(30), borderSide: BorderSide.none)))),
                const SizedBox(width: 10),
                Container(decoration: const BoxDecoration(color: Colors.deepPurpleAccent, shape: BoxShape.circle), child: IconButton(icon: const Icon(Icons.send_rounded, color: Colors.white), onPressed: _sendMessage)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}