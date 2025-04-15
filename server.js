const express = require('express');
const { PythonShell } = require('python-shell');
const cors = require('cors');
const app = express();

// Enable CORS
app.use(cors({
  origin: 'http://localhost:3000',
  methods: ['POST', 'GET']
}));

app.use(express.json());

// Prediction endpoint
app.post('/api/predict', async (req, res) => {

  console.log('Received formData:', req.body);

  try {
    const options = {
      mode: 'text',
      pythonPath: 'python',
      scriptPath: __dirname,
      args: [JSON.stringify(req.body)],
      timeout: 100000 // 100 seconds timeout
    };

    console.log('Sending to Python:', options.args[0]);

    const results = await new Promise((resolve, reject) => {
      PythonShell.run('predictor_service.py', options, (err, results) => {
        if (err){ 
          console.error('Python Error:', err);
          reject(err);
        }
        else {
          console.log('Python Returned:', results[0])
          resolve(results);
        }
      });
    });

    res.json(JSON.parse(results[0]));
  } catch (err) {
    console.error('Server error:', err);
    res.status(500).json({ 
      error: "Prediction failed",
      details: err.message,
      stack: err.stack 
    });
  }
});

const PORT = 5000;
app.listen(PORT, () => {
  console.log(`Server ready at http://localhost:${PORT}`);
});