/* 
* Molecular Visualization HTTP Server
* Copyright (C) 2011-2012 Cyrille Favreau <cyrille_favreau@hotmail.com>
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Library General Public
* License as published by the Free Software Foundation; either
* version 2 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Library General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* aint with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
* Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
*
*/

#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS

#include <lacewing.h>

#include <map>
#include <vector>
#include <time.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <fstream>

#include "../../../RaytracingEngine/tags/version-00.02.00/Consts.h"
#include "../../../RaytracingEngine/tags/version-00.02.00/PDBReader.h"
#include "../../../RaytracingEngine/tags/version-00.02.00/Cuda/CudaKernel.h"

#include "wininet.h" // for clearing URL cache DeleteUrlCacheEntry
#pragma comment(lib, "wininet.lib") // for clearing URL cache DeleteUrlCacheEntry

extern bool jo_write_jpg(const char *filename, const void *data, int width, int height, int comp, int quality);

// Requests
std::map<std::string,std::string> gRequests;

// ----------------------------------------------------------------------
// Kernel
// ----------------------------------------------------------------------
typedef CudaKernel GPUKERNEL;
GPUKERNEL* gpuKernel(nullptr);

// ----------------------------------------------------------------------
// Stats
// ----------------------------------------------------------------------
int gNbCalls(0);

// ----------------------------------------------------------------------
// Molecules
// ----------------------------------------------------------------------
size_t gCurrentProtein(0);
std::vector<std::string> gProteinNames;

// ----------------------------------------------------------------------
// Scene
// ----------------------------------------------------------------------
unsigned int gWindowWidth  = 512;
unsigned int gWindowHeight = gWindowWidth;
unsigned int gWindowDepth  = 4;

float4 gBkGrey  = {0.5f, 0.5f, 0.5f, 0.f};
float4 gBkBlack = {0.f, 0.f, 0.f, 0.f};
int   gTotalPathTracingIterations = 5;
int4  misc = {otJPEG,0,0,1};

SceneInfo gSceneInfo = 
{ 
   gWindowWidth,               // width
   gWindowHeight,              // height
   true,                       // shadowsEnabled
   10,                         // nbRayIterations
   3.f,                        // transparentColor
   1000000.f,                  // viewDistance
   0.6f,                       // shadowIntensity
   20.f,                       // width3DVision
   gBkGrey,                    // backgroundColor
   false,                      // supportFor3DVision
   false,                      // renderBoxes
   0,                          // pathTracingIteration
   gTotalPathTracingIterations,// maxPathTracingIterations
   misc                        // outputType
};

bool   gSceneHasChanged(true);
bool   gSpecular(true);
bool   gAnimate(false);
int    gTickCount(0);
float  gDefaultAtomSize(100.f);
float  gDefaultStickSize(80.f);
int    gMaxPathTracingIterations = gTotalPathTracingIterations;
int    gNbMaxBoxes( 8*8*8 );
int    gGeometryType(0);
int    gAtomMaterialType(0);
int    gBox(0);
float4 gRotationCenter = { 0.f, 0.f, 0.f, 0.f };

// Scene description and behavior
int gNbBoxes      = 0;
int gNbPrimitives = 0;
int gNbLamps      = 0;
int gNbMaterials  = 0;

// Camera information
float4 gViewPos    = { 0.f, 0.f, -15000.f, 0.f };
float4 gViewDir    = { 0.f, 0.f, -10000.f, 0.f };
float4 gViewAngles = { 0.f, 0.f, 0.f, 0.f };

// ----------------------------------------------------------------------
// Post processing
// ----------------------------------------------------------------------
PostProcessingInfo gPostProcessingInfo = 
{ 
   ppe_none, 
   1000.f, 
   5000.f, 
   60 
};

// ----------------------------------------------------------------------
// Utils
// ----------------------------------------------------------------------
void saturateFloat4(float4& value, const float min, const float max )
{
   value.x = (value.x < min) ? min : value.x;
   value.y = (value.y < min) ? min : value.y;
   value.z = (value.z < min) ? min : value.z;
   value.x = (value.x > max) ? max : value.x;
   value.y = (value.y > max) ? max : value.y;
   value.z = (value.z > max) ? max : value.z;
}

float4 readFloat4(const std::string value)
{
   float4 result = {0.f,0.f,0.f,0.f};
   std::string element;
   int i(0);
   for( int j(0); j<value.length(); ++j)
   {
      if( value[j] == ',' )
      {
         switch( i )
         {
         case 0: result.x = static_cast<float>(atof(element.c_str())); break;
         case 1: result.y = static_cast<float>(atof(element.c_str())); break;
         case 2: result.z = static_cast<float>(atof(element.c_str())); break;
         }
         element = "";
         i++;
      }
      else
      {
         element += value[j];
      }
   }
   if( element.length() != 0 )
   {
      result.z = static_cast<float>(atof(element.c_str()));
   }
   return result;
}

/*
________________________________________________________________________________

Create Random Materials
________________________________________________________________________________
*/
void createRandomMaterials()
{
   float4 specular;
   // Materials
   specular.z = 0.0f;
   specular.w = 1.0f;
   for( int i(0); i<NB_MAX_MATERIALS; ++i ) 
   {
      specular.x = (i>=0 && i<80 /*&& i%2==0*/) ? 0.5f : 0.f;
      specular.y = (i>=0 && i<80 /*&& i%2==0*/) ? 500.f: 10.f;
      specular.z = 0.f;
      specular.w = 0.1f;

      float innerIllumination = 0.f;
      float reflection   = 0.f;

      // Transparency & refraction
      float refraction = (i>=20 && i<80 && i%2==0) ? 1.33f : 0.f; 
      float transparency = (i>=20 && i<80 && i%2==0) ? 0.9f : 0.f; 

      int   textureId = NO_MATERIAL;
      float r,g,b;
      float noise = 0.f;
      bool  procedural = false;

      r = 0.5f+rand()%40/100.f;
      g = 0.5f+rand()%40/100.f;
      b = 0.5f+rand()%40/100.f;
      // Proteins
      switch( i%10 )
      {
      case  0: r = 0.8f;        g = 0.7f;        b = 0.7f;         break; 
      case  1: r = 0.7f;        g = 0.7f;        b = 0.7f;         break; // C Gray
      case  2: r = 174.f/255.f; g = 174.f/255.f; b = 233.f/255.f;  break; // N Blue
      case  3: r = 0.9f;        g = 0.4f;        b = 0.4f;         break; // O 
      case  4: r = 0.9f;        g = 0.9f;        b = 0.9f;         break; // H White
      case  5: r = 0.0f;        g = 0.5f;        b = 0.6f;         break; // B
      case  6: r = 0.5f;        g = 0.5f;        b = 0.7f;         break; // F Blue
      case  7: r = 0.8f;        g = 0.6f;        b = 0.3f;         break; // P
      case  8: r = 241.f/255.f; g = 196.f/255.f; b = 107.f/255.f;  break; // S Yellow
      case  9: r = 0.9f;        g = 0.3f;        b = 0.3f;         break; // V
      }

      switch(i)
      {
         // Wall materials
      case 80: r=127.f/255.f; g=127.f/255.f; b=127.f/255.f; specular.x = 0.2f; specular.y = 10.f; specular.w = 0.3f; break;
      case 81: r=154.f/255.f; g= 94.f/255.f; b= 64.f/255.f; specular.x = 0.1f; specular.y = 100.f; specular.w = 0.1f; break;
      case 82: r= 92.f/255.f; g= 93.f/255.f; b=150.f/255.f; break; 
      case 83: r = 100.f/255.f; g = 20.f/255.f; b = 10.f/255.f; break;

         // Lights
      case 95: r = 1.0f; g = 1.0f; b = 1.0f; refraction = 1.66f; transparency=0.9f; break;
      case 96: r = 1.0f; g = 1.0f; b = 1.0f; specular.x = 0.f; specular.y = 100.f; specular.w = 0.1f; reflection = 0.8f; break;
      case 97: r = 0.9f; g = 1.3f; b = 1.f; specular.x = 0.f; specular.y = 10.f; specular.w = 0.1f; /*textureId = 0;*/ break;
      //case 98: innerIllumination = 0.5f; break;
      case 99: r = 1.0f; g = 1.0f; b = 1.0f; innerIllumination = 1.f; break;
      }

      gNbMaterials = gpuKernel->addMaterial();
      gpuKernel->setMaterial( 
         gNbMaterials,
         r, g, b, noise,
         reflection, 
         refraction,
         procedural,
         false,0,
         transparency,
         textureId,
         specular.x, specular.y, specular.w, innerIllumination );
   }
}

// ----------------------------------------------------------------------
// Create 3D Scene
// ----------------------------------------------------------------------
float4 createScene( const std::string& fileName, const int structureType, const int scheme, const PostProcessingInfo& postProcessingInfo )
{
   // 3D Scene
   gpuKernel->setCamera( gViewPos, gViewDir, gViewAngles );

   // Lamp
   gNbPrimitives = gpuKernel->addPrimitive( ptSphere );
   gpuKernel->setPrimitive( gNbPrimitives, 0, 20000.f, 14000.f, -50000.f, 500.f, 0.f, 0.f, 99, 1 , 1);

   // PDB
   PDBReader prbReader;
   float4 size = prbReader.loadAtomsFromFile(
      fileName, *gpuKernel, 10, gNbMaxBoxes,
      static_cast<GeometryType>(structureType), 
      gDefaultAtomSize, gDefaultStickSize, scheme );
   gNbBoxes = gpuKernel->getNbActiveBoxes();

   float roomSize = fabs(size.x);
   roomSize = (fabs(size.y)>roomSize) ? fabs(size.y) : roomSize;
   roomSize = (fabs(size.z)>roomSize) ? fabs(size.z) : roomSize;
   roomSize *= 250.f;


#if 0
   if( postProcessingInfo.type.x != ppe_ambientOcclusion )
   {
      // Ground
      gNbPrimitives = gpuKernel->addPrimitive( ptXYPlane );  gpuKernel->setPrimitive( 
         gNbPrimitives, gNbBoxes+1,      
         0.f, 0.f, roomSize*0.6f, 
         gSceneInfo.viewDistance.x, gSceneInfo.viewDistance.x, 0.f, 
         83, 1, 1); 
   }
#endif // 0

   gNbBoxes = gpuKernel->compactBoxes();
   return size;
}

void initializeMolecules()
{
   // Proteins vector
	gProteinNames.push_back("3VM9");
   gProteinNames.push_back("1BNA");
   gProteinNames.push_back("3SUI");
   gProteinNames.push_back("1ACY");
   gProteinNames.push_back("3VHS");
	gProteinNames.push_back("4FMC");
	gProteinNames.push_back("3TGW");
	gProteinNames.push_back("4FI3");
	gProteinNames.push_back("3VJM");
	gProteinNames.push_back("4FME");
	gProteinNames.push_back("3U7D");
   gProteinNames.push_back("3U2Z");
	gProteinNames.push_back("3UA5");
	gProteinNames.push_back("3VKL");
	gProteinNames.push_back("3VKM");
}

static char encoding_table[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                                'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                                'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
                                'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                                'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                                'w', 'x', 'y', 'z', '0', '1', '2', '3',
                                '4', '5', '6', '7', '8', '9', '+', '/'};
static char *decoding_table = nullptr;
static int mod_table[] = {0, 2, 1};

char *base64_encode(const unsigned char *data,
                    size_t input_length,
                    size_t *output_length) 
{
    *output_length = (size_t) (4.0 * ceil((double) input_length / 3.0));

    char *encoded_data = (char*)malloc(*output_length+1);
    if (encoded_data == NULL) return NULL;

    for (int i = 0, j = 0; i < input_length;) {

        uint32_t octet_a = i < input_length ? data[i++] : 0;
        uint32_t octet_b = i < input_length ? data[i++] : 0;
        uint32_t octet_c = i < input_length ? data[i++] : 0;

        uint32_t triple = (octet_a << 0x10) + (octet_b << 0x08) + octet_c;

        encoded_data[j++] = encoding_table[(triple >> 3 * 6) & 0x3F];
        encoded_data[j++] = encoding_table[(triple >> 2 * 6) & 0x3F];
        encoded_data[j++] = encoding_table[(triple >> 1 * 6) & 0x3F];
        encoded_data[j++] = encoding_table[(triple >> 0 * 6) & 0x3F];
    }

    for (int i = 0; i < mod_table[input_length % 3]; i++)
    {
        encoded_data[*output_length - 1 - i] = '=';
    }

    encoded_data[*output_length] = 0;
    return encoded_data;
}

char* convertToBMP( char* buffer )
{
	unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,  0,0};
	unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0,  1,0, 24,0};
	unsigned char bmppad[3] = {0,0,0};

   int w = gWindowWidth;
	int h = gWindowHeight;
	int filesize = 54 + gWindowDepth*w*h;

	bmpfileheader[ 2] = (unsigned char)(filesize    );
	bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
	bmpfileheader[ 4] = (unsigned char)(filesize>>16);
	bmpfileheader[ 5] = (unsigned char)(filesize>>24);

	bmpinfoheader[ 4] = (unsigned char)(       w    );
	bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
	bmpinfoheader[ 6] = (unsigned char)(       w>>16);
	bmpinfoheader[ 7] = (unsigned char)(       w>>24);

	bmpinfoheader[ 8] = (unsigned char)(       h    );
	bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
	bmpinfoheader[10] = (unsigned char)(       h>>16);
	bmpinfoheader[11] = (unsigned char)(       h>>24);

   char* result = new char[filesize];
   memcpy(result   ,bmpfileheader,14);
   memcpy(result+14,bmpinfoheader,40);
   memcpy(result+54,buffer,gWindowDepth*w*h);
   return result;
}

// 
void onGet(Lacewing::Webserver &Webserver, Lacewing::Webserver::Request &request)
{
   float4 cameraOrigin = gViewPos;
   float4 cameraTarget = gViewDir;
   float4 cameraAngles = gViewAngles;

   float4 moleculeRotationAngles = {static_cast<float>(rand()%360),static_cast<float>(rand()%360),0.f,0.f};

   int structureType = rand()%5;
   int imageSize = 0;
   std::string moleculeId(gProteinNames[gCurrentProtein]);
   int scheme=rand()%3;
   SceneInfo sceneInfo(gSceneInfo);
   PostProcessingInfo postProcessingInfo(gPostProcessingInfo);
   //postProcessingInfo.type.x = (rand()%3==0) ? 2 : 0;

   // --------------------------------------------------------------------------------
   // Default values
   // --------------------------------------------------------------------------------
   
   if (!strcmp(request.URL(), "get"))
   {
      std::string requestStr;
      try
      {
         Lacewing::Webserver::Request::Parameter* p=request.GET();
         requestStr += request.GetAddress().ToString();
         requestStr += ": ";
         requestStr += request.URL();
         requestStr += "?";
         while( p != nullptr )
         {
            requestStr += p->Name();
            requestStr += "=";
            requestStr += p->Value();
            if( strcmp(p->Name(),"molecule")==0 )
            {
               // --------------------------------------------------------------------------------
               // Molecule
               // --------------------------------------------------------------------------------
               moleculeId = p->Value();
            }
            else if ( strcmp(p->Name(),"rotation") == 0 )
            {
               // --------------------------------------------------------------------------------
               // rotation angles
               // --------------------------------------------------------------------------------
               moleculeRotationAngles = readFloat4(p->Value());
               moleculeRotationAngles.x = moleculeRotationAngles.x/180.f*static_cast<float>(M_PI);
               moleculeRotationAngles.y = moleculeRotationAngles.y/180.f*static_cast<float>(M_PI);
               moleculeRotationAngles.z = moleculeRotationAngles.z/180.f*static_cast<float>(M_PI);
            }
            else if ( strcmp(p->Name(),"bkcolor") == 0 )
            {
               // --------------------------------------------------------------------------------
               // Backgroud color
               // --------------------------------------------------------------------------------
               sceneInfo.backgroundColor = readFloat4(p->Value());
               sceneInfo.backgroundColor.x /= 255.f;
               sceneInfo.backgroundColor.y /= 255.f;
               sceneInfo.backgroundColor.z /= 255.f;
               saturateFloat4(sceneInfo.backgroundColor,0.f,255.f);
            }
            else if ( strcmp(p->Name(),"structure") == 0 )
            {
               // --------------------------------------------------------------------------------
               // structure
               // --------------------------------------------------------------------------------
               structureType = atoi(p->Value());
               if( structureType<0 || structureType>4 ) structureType = 0;
            }
            else if ( strcmp(p->Name(),"scheme") == 0 )
            {
               // --------------------------------------------------------------------------------
               // scheme
               // --------------------------------------------------------------------------------
               scheme = atoi(p->Value());
               if( scheme<0 || scheme>2 ) scheme = 0;
            }
            else if ( strcmp(p->Name(),"quality") == 0 )
            {
               // --------------------------------------------------------------------------------
               // Quality
               // --------------------------------------------------------------------------------
               sceneInfo.maxPathTracingIterations.x = atoi(p->Value());
               sceneInfo.maxPathTracingIterations.x = (sceneInfo.maxPathTracingIterations.x>20) ? 20 : sceneInfo.maxPathTracingIterations.x;
            }
            else if ( strcmp(p->Name(),"size") == 0 )
            {
               // --------------------------------------------------------------------------------
               // Image Size
               // --------------------------------------------------------------------------------
               imageSize = atoi(p->Value());
               switch( imageSize ) 
               {
               default: gWindowWidth=768;  gWindowHeight=768;
               case  1: gWindowWidth=1024; gWindowHeight=1024; break;
               case  2: gWindowWidth=1600; gWindowHeight=1600; break;
               case  3: gWindowWidth=1920; gWindowHeight=1920; break;
               case  4: gWindowWidth=2048; gWindowHeight=2048; break;
               }
               sceneInfo.width.x  = gWindowWidth;
               sceneInfo.height.x = gWindowHeight;
            }
            else if ( strcmp(p->Name(),"postprocessing") == 0 )
            {
               // --------------------------------------------------------------------------------
               // structure
               // --------------------------------------------------------------------------------
               int postProcessing = atoi(p->Value());
               if( postProcessing<0 || postProcessing>2 ) postProcessing = 0;
               postProcessingInfo.type.x = postProcessing;
            }

            p = p->Next();
            if(p != nullptr) requestStr += "&";
         }
         std::cout << requestStr << std::endl;

         // --------------------------------------------------------------------------------
         // PDB File management
         // --------------------------------------------------------------------------------
         std::string fileName("../Pdb/");
         std::string moleculeName;
         moleculeName += ( moleculeId.length() == 0 ) ? gProteinNames[gCurrentProtein] : moleculeId;
         moleculeName += ".pdb";

         fileName += moleculeName;

         // Check file existence
         std::ifstream file( fileName.c_str() );
         if( file.is_open() )
         {
            file.close();
         }
         else
         {
            // If file is not in the cache, download it

            std::string url("http://www.rcsb.org/pdb/files/");
            url += moleculeName;
            HINTERNET IntOpen = ::InternetOpen("Sample", LOCAL_INTERNET_ACCESS, NULL, 0, 0);
            HINTERNET handle = ::InternetOpenUrl(IntOpen, url.c_str(), NULL, NULL, NULL, NULL);

            if( handle )
            {
               std::ofstream myfile(fileName);
               if (myfile.is_open())
               {
                  request << "<p align=center>PDB File was not in the cache and had to be downloaded from <a href=http://www.rcsb.org>Protein Data Bank</a></p>";
                  char buffer[2];
                  DWORD dwRead=0;
                  while(::InternetReadFile(handle, buffer, sizeof(buffer)-1, &dwRead) == TRUE)
                  {
                     if ( dwRead == 0) 
                        break;
                     myfile << buffer;
                  }
                  myfile.close();
               }
            }
            else
            {
               // TODO!!!!
               request << "<p align=center>Unknown molecule</p>";
            }
            ::InternetCloseHandle(handle);   
         }

         // --------------------------------------------------------------------------------
         // Create 3D Scene
         // --------------------------------------------------------------------------------
         size_t len(gWindowWidth*gWindowHeight*gWindowDepth);
	      char* image = new char[len];
         if( image != nullptr )
         {
            long renderingTime = GetTickCount();
            memset( image, 0, len );
            gpuKernel = new GPUKERNEL(false, true);
            gSceneInfo.pathTracingIteration.x = 0;
            gpuKernel->setSceneInfo( sceneInfo );
            gpuKernel->initBuffers();

            createRandomMaterials();

            // Bkground Color
            gpuKernel->setMaterial( 83, sceneInfo.backgroundColor.x, sceneInfo.backgroundColor.y, sceneInfo.backgroundColor.z, 0.f,
               0.f, 0.f, false, false, 0, 0.f, NO_TEXTURE, 0.5f, 100.f, 0.f, 0.f );


            float4 size = createScene( fileName, structureType, scheme, postProcessingInfo );
            cameraTarget.z = -size.z*250.f;
            cameraOrigin.z = cameraTarget.z-4000.f;

            // Post processing effects
            postProcessingInfo.param1.x = -cameraTarget.z;
            postProcessingInfo.param2.x = (postProcessingInfo.type.x==0) ? 
               sceneInfo.maxPathTracingIterations.x*10.f : 5000.f;
            postProcessingInfo.param3.x = (postProcessingInfo.type.x != 2 ) ? 40+sceneInfo.maxPathTracingIterations.x*5 : 16;

            // Shadows
            sceneInfo.shadowsEnabled.x = (postProcessingInfo.type.x != 2);

            // Rotation
            gpuKernel->rotatePrimitives( gRotationCenter, moleculeRotationAngles, 10, gNbBoxes );

            // Background color
            sceneInfo.backgroundColor = (postProcessingInfo.type.x == 2 ) ? gBkBlack : sceneInfo.backgroundColor;

            // Rendering process
            for( int i(0); i<sceneInfo.maxPathTracingIterations.x; ++i)
            {
               sceneInfo.pathTracingIteration.x = i;
               gpuKernel->setPostProcessingInfo( postProcessingInfo );
               gpuKernel->setSceneInfo( sceneInfo );
               gpuKernel->setCamera( cameraOrigin, cameraTarget, cameraAngles );
               gpuKernel->render_begin(0.f);
               gpuKernel->render_end((char*)image);
            }


            std::string jpgName = moleculeId + ".jpg";
            size_t len;
            char* buffer = nullptr;
            long bufferLength;
            jo_write_jpg(jpgName.c_str(),image,sceneInfo.width.x,sceneInfo.height.x,3,100);
            FILE * pFile;
            size_t result;

            pFile = fopen ( jpgName.c_str(), "rb" );
            if (pFile!=NULL) 
            {
               // obtain file size:
               fseek (pFile , 0 , SEEK_END);
               bufferLength = ftell (pFile);
               rewind (pFile);

               // allocate memory to contain the whole file:
               buffer = new char[bufferLength];

               // copy the file into the buffer:
               result = fread (buffer,1,bufferLength,pFile);
               if (result != bufferLength) 
               {
                  //fputs ("Reading error",stderr); exit (3);
               }

               /* the whole file is now loaded in the memory buffer. */

               // terminate
               fclose (pFile);
            }

#if 0
            request << "<body>";
            request << "<p align=\"center\"><b>Molecule:</b>" << moleculeId.c_str() << "</p>";
            request << "<p align=\"center\"><img border=5 bgcolor=#000000 src=\"data:image/jpg;base64,";
            request << base64_encode( (const unsigned char*)buffer, bufferLength, &len );
            request << "\"/></p>";
            request << "<p align=\"center\">Copyright(C) Cyrille Favreau</p>";
            renderingTime = GetTickCount()-renderingTime;
            request << "<p align=\"center\">Rendering time: " << renderingTime << " milliseconds on nVidia GTX 480</p>";
            /*
            request << "<p align=\"center\">molecule=XXXX 4 capital letters identifiying the molecule. The list bellow is  the only one you can use for now).<br/>";
            request << "<p align=\"center\">scheme=[0|1|2] 0: Standard, 1: Chain, 2: Residue<br/>";
            request << "<p align=\"center\">structure=[0|1|2|3] 0: Real size atoms, 1: Fixed size atoms, 2: Sticks, 3: Sticks and atoms<br/>";
            request << "<p align=\"center\">rotation=[x,y,z] Rotates the molecule according to x,y and z. Note that x,y,and z are real numbers specifying degrees of rotation for each axe.<br/>";
            request << "<p align=\"center\">quality=[1-100] Identifies the number of iterations to process. The higher the better, and slower...<br/>";
            request << "<p align=\"center\">bkcolor=[r,g,b] Specifies the red, green and blue values for background color(example: bkcolor=255,0,127)<br/>";
            request << "<p align=\"center\">postprocessing=[0|1|2] 0: None, 1: Depth of field, 2: Ambient occlusion</p>";
            request << "<p align=\"center\">Syntax: http://molecular-visualization.no-ip.org/get?molecule=XXXX[&scheme=0|1|2][&structure=0|1|2|3][&rotation=float,float,\<float\>][&quality=integer]<br/>";
            request << "<p align=\"center\">Example: http://molecular-visualization.no-ip.org/get?postprocessing=0&bkcolor=120,120,120&quality=1000&rotation=0,0,0&molecule=2M1L</p>";
            */
            request << "<p align=\"center\">Help: <a href=\"http://cudaopencl.blogspot.com\">http://cudaopencl.blogspot.com</a></p>";
            request << "<p align=\"center\"><a href=\"http://www.molecular-visualization.com\">http://www.molecular-visualization.com</a></p>";
            request << "</body>";
            delete [] buffer;
#else
            request << "data:image/jpg;base64,";
            request << base64_encode( (const unsigned char*)buffer, bufferLength, &len );
            request.AddHeader("Access-Control-Allow-Origin", "*"); // Needed by Chrome!!
            delete [] buffer;
#endif // 0

            delete gpuKernel;
            delete image;
         }

         gCurrentProtein++;
         gCurrentProtein = gCurrentProtein%gProteinNames.size();
      }
      catch(...)
      {
         request << "An exception occured :-( Please try again";
      }
      gRequests[request.GetAddress().ToString()] = requestStr;
      gNbCalls++;
   }
   else
   {
      request << gNbCalls << " calls so far<br/>";
      std::map<std::string,std::string>::const_iterator iter = gRequests.begin();
      while( iter != gRequests.end() )
      {
         request << (*iter).second.c_str() << "<br/>";
         ++iter;
      }
   }
}

int main(int argc, char * argv[])
{
   Lacewing::EventPump EventPump;
   Lacewing::Webserver Webserver(EventPump);

   Webserver.onGet(onGet);
   Webserver.Host(8083);    

   initializeMolecules();

   EventPump.StartEventLoop();

   return 0;
}
